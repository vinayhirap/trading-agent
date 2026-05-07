# trading-agent/src/prediction/rl_position_sizer.py
"""
Reinforcement Learning Position Sizer — DQN Architecture

Why RL for position sizing:
  Fixed risk % (1% per trade) is suboptimal — it ignores:
  - Current market regime (size up in trends, down in chop)
  - Recent performance (reduce size after losses = drawdown control)
  - Signal strength (size proportional to confidence)
  - Volatility state (ATR × sizing = consistent risk)

DQN State space (12 features):
  1. Current regime (0-3: bear/ranging-h/ranging-l/bull)
  2. Signal confidence (0.5 - 0.95)
  3. ATR ratio (current vol / avg vol)
  4. RSI (normalized 0-1)
  5. Recent win rate (last 10 trades)
  6. Current drawdown % from peak
  7. Days since last loss
  8. Portfolio utilization % (positions / max)
  9. Market session quality (0-1)
  10. FII flow signal (-1, 0, +1)
  11. Consecutive wins/losses streak
  12. Sharpe ratio rolling 20 trades

Action space (7 discrete actions):
  0: 0.0% (skip trade)
  1: 0.5% capital
  2: 1.0% capital (default)
  3: 1.5% capital
  4: 2.0% capital
  5: 2.5% capital
  6: 3.0% capital (max — only in STRONG signals)

Training:
  - Uses paper trade history as experience replay
  - Reward = trade P&L normalized by risk taken
  - Trains every 50 completed trades
  - Model saved to models/rl_sizer.pkl

Usage:
    from src.prediction.rl_position_sizer import RLPositionSizer
    sizer = RLPositionSizer(capital=10000)
    size_pct = sizer.get_size(state_dict)
    position_value = capital * size_pct
"""
from __future__ import annotations
import json
import pickle
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "rl_sizer.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

# ── Action space ──────────────────────────────────────────────────────────────
ACTIONS = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
N_ACTIONS = len(ACTIONS)
STATE_DIM = 12

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA          = 0.95    # discount factor
LR             = 0.001   # learning rate
EPSILON_START  = 1.0     # exploration start
EPSILON_END    = 0.05    # minimum exploration
EPSILON_DECAY  = 0.995   # decay per episode
BATCH_SIZE     = 32
MEMORY_SIZE    = 2000
TARGET_UPDATE  = 50      # update target network every N steps
MIN_MEMORY     = 100     # minimum experiences before training


# ── Neural network (pure numpy — no PyTorch dependency) ───────────────────────

class SimpleNN:
    """
    2-layer neural network implemented in pure numpy.
    No PyTorch/TF dependency — runs anywhere.

    Architecture: 12 → 64 → 32 → 7
    Activation: ReLU hidden, linear output
    """

    def __init__(self, input_dim: int = STATE_DIM, output_dim: int = N_ACTIONS):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, 64)  * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 32)         * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, output_dim) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        h1 = np.maximum(0, x @ self.W1 + self.b1)    # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)   # ReLU
        return h2 @ self.W3 + self.b3                 # Linear output

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_weights(self) -> dict:
        return {"W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "W3": self.W3, "b3": self.b3}

    def set_weights(self, weights: dict):
        self.W1 = weights["W1"]; self.b1 = weights["b1"]
        self.W2 = weights["W2"]; self.b2 = weights["b2"]
        self.W3 = weights["W3"]; self.b3 = weights["b3"]

    def update(self, x: np.ndarray, target_q: np.ndarray, lr: float = LR):
        """Single gradient step using MSE loss."""
        x = np.atleast_2d(x)

        # Forward
        h1      = np.maximum(0, x @ self.W1 + self.b1)
        h2      = np.maximum(0, h1 @ self.W2 + self.b2)
        q_vals  = h2 @ self.W3 + self.b3

        # Loss gradient (MSE)
        delta3 = 2 * (q_vals - target_q) / len(x)

        # Backprop layer 3
        dW3 = h2.T @ delta3
        db3 = delta3.sum(axis=0)
        delta2 = delta3 @ self.W3.T * (h2 > 0)

        # Backprop layer 2
        dW2 = h1.T @ delta2
        db2 = delta2.sum(axis=0)
        delta1 = delta2 @ self.W2.T * (h1 > 0)

        # Backprop layer 1
        dW1 = x.T @ delta1
        db1 = delta1.sum(axis=0)

        # Gradient descent
        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1


# ── Experience replay ─────────────────────────────────────────────────────────

@dataclass
class Experience:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """DQN agent for position sizing decisions."""

    def __init__(self):
        self.q_network      = SimpleNN()
        self.target_network = SimpleNN()
        self.target_network.set_weights(self.q_network.get_weights())

        self.memory   = deque(maxlen=MEMORY_SIZE)
        self.epsilon  = EPSILON_START
        self.step_count = 0
        self.episode_count = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        q_vals = self.q_network.predict(state)[0]
        return int(np.argmax(q_vals))

    def store(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """Sample batch and train. Returns loss or None if insufficient data."""
        if len(self.memory) < MIN_MEMORY:
            return None

        batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))

        states      = np.array([e.state      for e in batch])
        actions     = np.array([e.action     for e in batch])
        rewards     = np.array([e.reward     for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones       = np.array([e.done       for e in batch], dtype=float)

        # Compute target Q-values
        current_q = self.q_network.predict(states)
        next_q    = self.target_network.predict(next_states)
        target_q  = current_q.copy()

        for i in range(len(batch)):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + GAMMA * next_q[i].max()

        # Update
        self.q_network.update(states, target_q)

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.step_count += 1

        # Sync target network
        if self.step_count % TARGET_UPDATE == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        # Return MSE loss
        pred_q = self.q_network.predict(states)
        return float(np.mean((pred_q - target_q) ** 2))

    def save(self, path: Path = MODEL_PATH):
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "q_weights":  self.q_network.get_weights(),
                    "epsilon":    self.epsilon,
                    "step_count": self.step_count,
                }, f)
        except Exception as e:
            logger.warning(f"RL sizer save failed: {e}")

    def load(self, path: Path = MODEL_PATH) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.q_network.set_weights(data["q_weights"])
            self.target_network.set_weights(data["q_weights"])
            self.epsilon    = data.get("epsilon", EPSILON_END)
            self.step_count = data.get("step_count", 0)
            logger.info(f"RL sizer loaded: ε={self.epsilon:.3f}, steps={self.step_count}")
            return True
        except Exception as e:
            logger.warning(f"RL sizer load failed: {e}")
            return False


# ── State builder ─────────────────────────────────────────────────────────────

class StateBuilder:
    """Converts raw market data into normalized DQN state vector."""

    REGIME_MAP = {
        "BEAR_TREND":   0.0,
        "RANGING_HIGH": 0.25,
        "RANGING_LOW":  0.5,
        "BULL_TREND":   1.0,
    }

    def build(self, context: dict) -> np.ndarray:
        """
        Build normalized state vector from context dict.

        context keys:
            regime_str:     "BULL_TREND" etc.
            confidence:     float 0-1
            atr_ratio:      float (current/avg ATR)
            rsi:            float 0-100
            win_rate_10:    float 0-1 (last 10 trades)
            drawdown_pct:   float 0-1 (current DD from peak)
            days_since_loss:int
            portfolio_util: float 0-1
            session_quality:float 0-1
            fii_signal:     float -1/0/1
            streak:         int (positive=wins, negative=losses)
            sharpe_20:      float (rolling Sharpe last 20 trades)
        """
        def _clip(v, lo=0.0, hi=1.0, default=0.5):
            try:
                return float(np.clip(float(v), lo, hi))
            except (TypeError, ValueError):
                return default

        regime_str = context.get("regime_str", "RANGING_LOW")
        regime_val = self.REGIME_MAP.get(regime_str, 0.5)

        confidence     = _clip(context.get("confidence", 0.55), 0.5, 1.0)
        atr_ratio      = _clip(context.get("atr_ratio", 1.0),   0.2, 4.0) / 4.0
        rsi            = _clip(context.get("rsi", 50),          0, 100)   / 100.0
        win_rate_10    = _clip(context.get("win_rate_10", 0.5))
        drawdown_pct   = _clip(context.get("drawdown_pct", 0.0), 0, 0.20) / 0.20
        days_loss      = _clip(context.get("days_since_loss", 5), 0, 30)  / 30.0
        portfolio_util = _clip(context.get("portfolio_util", 0.0))
        session_q      = _clip(context.get("session_quality", 0.7))
        fii_signal     = _clip(context.get("fii_signal", 0.0), -1, 1, 0) / 2.0 + 0.5
        streak_raw     = int(context.get("streak", 0))
        streak_norm    = _clip((streak_raw + 10) / 20.0)  # normalize -10..+10 → 0..1
        sharpe_20      = _clip((context.get("sharpe_20", 0) + 2) / 4.0)  # -2..+2 → 0..1

        return np.array([
            regime_val, confidence, atr_ratio, rsi,
            win_rate_10, drawdown_pct, days_loss, portfolio_util,
            session_q, fii_signal, streak_norm, sharpe_20,
        ], dtype=np.float32)


# ── Reward function ───────────────────────────────────────────────────────────

def compute_reward(
    trade_pnl_pct:  float,   # actual trade return %
    size_taken:     float,   # position size % used
    risk_taken:     float,   # stop loss distance %
    was_correct:    bool,    # signal was right direction
) -> float:
    """
    Reward = risk-adjusted return.
    Penalizes taking big size on wrong signals.
    Rewards taking big size on correct signals.
    """
    # Base reward: P&L relative to risk taken
    if risk_taken > 0:
        reward = trade_pnl_pct / risk_taken   # risk-adjusted
    else:
        reward = trade_pnl_pct * 10

    # Penalty for large size on wrong signal
    if not was_correct and size_taken > 0.015:
        reward -= size_taken * 20   # penalize large wrong bets

    # Bonus for large size on correct signal
    if was_correct and size_taken > 0.015:
        reward += size_taken * 10

    # Clip reward
    return float(np.clip(reward, -5.0, 5.0))


# ── Main RLPositionSizer ──────────────────────────────────────────────────────

class RLPositionSizer:
    """
    Public API for RL-based position sizing.

    Usage:
        sizer = RLPositionSizer(capital=10000)

        # Get position size for a trade
        size_pct = sizer.get_size({
            "regime_str":   "BULL_TREND",
            "confidence":   0.72,
            "atr_ratio":    1.2,
            "rsi":          58,
            "win_rate_10":  0.6,
            "drawdown_pct": 0.02,
        })
        # size_pct e.g. 0.015 = 1.5% of capital

        # After trade closes, record outcome
        sizer.record_outcome(size_pct, pnl_pct=0.023, correct=True, sl_pct=0.015)

        # Train every 50 trades
        sizer.maybe_train()
    """

    def __init__(self, capital: float = 10_000):
        self.capital      = capital
        self._agent       = DQNAgent()
        self._builder     = StateBuilder()
        self._loaded      = self._agent.load()
        self._last_state  = None
        self._last_action = None
        self._trade_count = 0
        self._stats       = self._load_stats()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_size(self, context: dict, greedy: bool = True) -> float:
        """
        Get position size as fraction of capital.
        Returns value in ACTIONS list: 0.0 to 0.030.

        greedy=True uses best known action (no exploration).
        greedy=False uses ε-greedy (for training).
        """
        state  = self._builder.build(context)
        action = self._agent.select_action(state, greedy=greedy)
        size   = ACTIONS[action]

        # Safety override: never take > 2% in RANGING_HIGH
        regime = context.get("regime_str", "")
        if regime == "RANGING_HIGH" and size > 0.015:
            size   = 0.010
            action = ACTIONS.index(0.010)

        # Safety override: never trade below confidence gate
        conf = float(context.get("confidence", 0.55))
        if conf < 0.55:
            size   = 0.0
            action = 0

        self._last_state  = state
        self._last_action = action

        logger.debug(f"RL sizer: action={action} → {size:.1%} | ε={self._agent.epsilon:.3f}")
        return size

    def get_size_rupees(self, context: dict) -> float:
        """Returns position value in rupees."""
        return self.capital * self.get_size(context)

    def record_outcome(
        self,
        size_taken:  float,
        pnl_pct:     float,
        correct:     bool,
        sl_pct:      float = 0.015,
        done:        bool  = True,
    ):
        """
        Call after a trade closes to store experience.
        The sizer learns from actual outcomes.
        """
        if self._last_state is None or self._last_action is None:
            return

        reward = compute_reward(pnl_pct, size_taken, sl_pct, correct)

        # Build next state (simplified — use terminal state if done)
        next_state = np.zeros(STATE_DIM, dtype=np.float32) if done else self._last_state

        self._agent.store(
            self._last_state, self._last_action,
            reward, next_state, done,
        )

        # Update stats
        self._trade_count += 1
        self._stats["total_trades"] += 1
        self._stats["total_reward"] += reward
        self._stats["recent_rewards"].append(reward)
        if len(self._stats["recent_rewards"]) > 50:
            self._stats["recent_rewards"] = self._stats["recent_rewards"][-50:]

        self._save_stats()
        self._last_state = None
        self._last_action = None

    def maybe_train(self, every_n: int = 50) -> Optional[float]:
        """
        Train if enough new experiences have accumulated.
        Call after every trade close.
        Returns loss or None.
        """
        if self._trade_count % every_n == 0 and self._trade_count > 0:
            loss = self._agent.train_step()
            if loss is not None:
                logger.info(
                    f"RL sizer trained: loss={loss:.4f} | "
                    f"ε={self._agent.epsilon:.3f} | "
                    f"trades={self._trade_count}"
                )
                self._agent.save()
            return loss
        return None

    def force_train(self, n_steps: int = 100) -> list[float]:
        """Force training for n steps. Use for initial warm-up."""
        losses = []
        for _ in range(n_steps):
            loss = self._agent.train_step()
            if loss is not None:
                losses.append(loss)
        if losses:
            self._agent.save()
            logger.info(f"RL forced train: {len(losses)} steps, avg loss={np.mean(losses):.4f}")
        return losses

    def get_stats(self) -> dict:
        return {
            "trained":         self._loaded,
            "epsilon":         round(self._agent.epsilon, 3),
            "step_count":      self._agent.step_count,
            "memory_size":     len(self._agent.memory),
            "total_trades":    self._stats["total_trades"],
            "avg_reward":      round(
                np.mean(self._stats["recent_rewards"]) if self._stats["recent_rewards"] else 0, 3
            ),
            "action_dist":     self._get_action_distribution(),
        }

    def _get_action_distribution(self) -> dict:
        """What sizes is the model recommending most?"""
        if len(self._agent.memory) < 10:
            return {}
        actions = [e.action for e in list(self._agent.memory)[-100:]]
        from collections import Counter
        counts = Counter(actions)
        return {f"{ACTIONS[a]:.1%}": c for a, c in sorted(counts.items())}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_stats(self) -> dict:
        stats_path = ROOT / "data" / "rl_sizer_stats.json"
        try:
            if stats_path.exists():
                return json.loads(stats_path.read_text())
        except Exception:
            pass
        return {"total_trades": 0, "total_reward": 0, "recent_rewards": []}

    def _save_stats(self):
        stats_path = ROOT / "data" / "rl_sizer_stats.json"
        try:
            stats_path.parent.mkdir(exist_ok=True)
            stats_path.write_text(json.dumps(self._stats, indent=2))
        except Exception:
            pass

    # ── Warm-up from trade history ────────────────────────────────────────────

    def warm_up_from_history(self, trade_history: list[dict]) -> int:
        """
        Bootstrap RL from existing paper trade history.
        trade_history: list of trade dicts from PaperBroker.get_trade_history()
        Returns number of experiences loaded.
        """
        loaded = 0
        for trade in trade_history:
            try:
                context = {
                    "confidence":   trade.get("confidence", 0.55),
                    "regime_str":   trade.get("regime", "RANGING_LOW"),
                    "atr_ratio":    1.0,
                    "rsi":          trade.get("rsi", 50),
                    "win_rate_10":  0.5,
                    "drawdown_pct": 0.0,
                    "session_quality": 0.7,
                }
                state  = self._builder.build(context)
                size   = float(trade.get("size_pct", 0.01))
                action = min(range(N_ACTIONS), key=lambda i: abs(ACTIONS[i] - size))
                pnl    = float(trade.get("pnl_pct", 0))
                correct= pnl > 0
                reward = compute_reward(pnl, size, 0.015, correct)

                self._agent.store(state, action, reward, state * 0, True)
                loaded += 1
            except Exception:
                continue

        logger.info(f"RL warm-up: loaded {loaded} experiences from history")
        return loaded


# ── Dashboard widget ──────────────────────────────────────────────────────────

def render_rl_widget():
    """
    Embed in System Health or create dedicated page:
        from src.prediction.rl_position_sizer import render_rl_widget
        render_rl_widget()
    """
    import streamlit as st
    import plotly.graph_objects as go

    st.subheader("🧠 RL Position Sizer")
    st.caption(
        "DQN agent learns optimal position sizing from trade outcomes. "
        "Pure numpy — no PyTorch needed. Trains every 50 trades."
    )

    sizer = RLPositionSizer()
    stats = sizer.get_stats()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model Loaded",  "✅ Yes" if stats["trained"] else "⚠️ Untrained")
    k2.metric("Exploration ε", f"{stats['epsilon']:.3f}",
              "1.0=random, 0.05=optimal")
    k3.metric("Training Steps",stats["step_count"])
    k4.metric("Avg Reward",    f"{stats['avg_reward']:+.3f}")

    # Action distribution
    dist = stats.get("action_dist", {})
    if dist:
        st.subheader("Position Size Distribution (last 100 decisions)")
        fig = go.Figure(go.Bar(
            x=list(dist.keys()),
            y=list(dist.values()),
            marker_color="#4da6ff",
            text=list(dist.values()),
            textposition="outside",
        ))
        fig.update_layout(
            height=220, template="plotly_dark",
            title="How often each size is recommended",
            xaxis_title="Position Size", yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Simulate sizing
    st.subheader("Size Simulator")
    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        sim_regime = st.selectbox("Regime", ["BULL_TREND","RANGING_LOW","RANGING_HIGH","BEAR_TREND"], key="rl_regime")
        sim_conf   = st.slider("Confidence", 0.50, 0.95, 0.65, 0.05, key="rl_conf")
    with sim2:
        sim_atr    = st.slider("ATR Ratio", 0.5, 3.0, 1.0, 0.1, key="rl_atr")
        sim_wr     = st.slider("Win Rate (10)", 0.2, 0.9, 0.5, 0.05, key="rl_wr")
    with sim3:
        sim_dd     = st.slider("Drawdown %", 0.0, 0.15, 0.02, 0.01, key="rl_dd")
        sim_streak = st.slider("Streak", -5, 5, 0, 1, key="rl_streak")

    context = {
        "regime_str": sim_regime, "confidence": sim_conf,
        "atr_ratio": sim_atr, "rsi": 55,
        "win_rate_10": sim_wr, "drawdown_pct": sim_dd,
        "session_quality": 0.7, "fii_signal": 0,
        "streak": sim_streak, "sharpe_20": 0.5,
    }
    size = sizer.get_size(context, greedy=True)
    size_rs = size * 10000

    color = "#00cc66" if size > 0.01 else "#ffaa00" if size > 0 else "#ff4444"
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:12px;text-align:center">'
        f'<div style="font-size:28px;font-weight:700;color:{color}">{size:.1%}</div>'
        f'<div style="color:#888;font-size:13px">of capital = ₹{size_rs:,.0f} on ₹10,000</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Training controls
    st.divider()
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🏃 Warm-up from trade history", key="rl_warmup"):
            with st.spinner("Loading trade history..."):
                try:
                    from src.execution.paper_broker import PaperBroker
                    from config.settings import settings
                    broker = PaperBroker(initial_capital=settings.INITIAL_CAPITAL)
                    history = broker.get_trade_history()
                    n = sizer.warm_up_from_history(history)
                    st.success(f"Loaded {n} experiences from {len(history)} trades")
                except Exception as e:
                    st.error(f"Warm-up failed: {e}")
    with col_t2:
        if st.button("🔧 Force train (100 steps)", key="rl_train"):
            with st.spinner("Training..."):
                losses = sizer.force_train(100)
                if losses:
                    st.success(f"Trained {len(losses)} steps | avg loss: {np.mean(losses):.4f}")
                else:
                    st.warning("Need more experiences first. Run warm-up.")