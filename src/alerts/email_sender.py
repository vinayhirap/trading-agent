# trading-agent/src/alerts/email_sender.py
"""
Email Sender — delivers alert notifications via SMTP.

Supports Gmail (app password), Outlook, and any SMTP server.
Configure in .env:
  ALERT_EMAIL_FROM     = you@gmail.com
  ALERT_EMAIL_PASSWORD = your-app-password   # Gmail: 16-char app password
  ALERT_EMAIL_TO       = you@gmail.com       # can be same as FROM
  ALERT_SMTP_HOST      = smtp.gmail.com      # default
  ALERT_SMTP_PORT      = 587                 # default (TLS)

Gmail setup:
  1. Enable 2FA on your Google account
  2. Go to myaccount.google.com → Security → App passwords
  3. Generate a 16-char password and put it in ALERT_EMAIL_PASSWORD
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from loguru import logger


class EmailSender:
    """
    Sends alert notification emails.
    Fails silently if not configured — alerts still work via UI.
    """

    def __init__(
        self,
        from_addr:  str,
        password:   str,
        to_addr:    str,
        smtp_host:  str = "smtp.gmail.com",
        smtp_port:  int = 587,
    ):
        self.from_addr = from_addr
        self.password  = password
        self.to_addr   = to_addr
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self._enabled  = bool(from_addr and password and to_addr)

        if self._enabled:
            logger.info(f"EmailSender ready: {from_addr} → {to_addr}")
        else:
            logger.info("EmailSender: not configured (email alerts disabled)")

    @property
    def is_configured(self) -> bool:
        return self._enabled

    def send_alert(self, fire) -> bool:
        """
        Send an alert notification email.
        fire: AlertFire dataclass or dict with .message, .symbol, .alert_type
        """
        if not self._enabled:
            return False

        try:
            subject, body = self._format(fire)
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = self.from_addr
            msg["To"]      = self.to_addr

            # Plain text part
            msg.attach(MIMEText(body["text"], "plain"))
            # HTML part
            msg.attach(MIMEText(body["html"], "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.from_addr, self.password)
                server.sendmail(self.from_addr, self.to_addr, msg.as_string())

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Email failed: {e}")
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Test SMTP connection. Returns (success, message)."""
        if not self._enabled:
            return False, "Email not configured. Add ALERT_EMAIL_FROM/PASSWORD/TO to .env"
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.from_addr, self.password)
            return True, f"✅ Connected to {self.smtp_host}:{self.smtp_port}"
        except smtplib.SMTPAuthenticationError:
            return False, "❌ Authentication failed. Check your app password."
        except smtplib.SMTPConnectError:
            return False, f"❌ Cannot connect to {self.smtp_host}:{self.smtp_port}"
        except Exception as e:
            return False, f"❌ Error: {e}"

    def send_test_email(self) -> tuple[bool, str]:
        """Send a test email to verify everything works end-to-end."""
        if not self._enabled:
            return False, "Email not configured."
        try:
            subject = "✅ AI Trading Agent — Email Alerts Working"
            now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            text    = f"Test email from AI Trading Agent\nSent at: {now}\nAlerts are configured correctly."
            html    = f"""
            <html><body style="font-family:sans-serif;background:#1a1a2e;color:#eee;padding:20px">
              <div style="max-width:500px;margin:0 auto;background:#16213e;border-radius:8px;padding:24px">
                <h2 style="color:#00cc66">✅ Email Alerts Working</h2>
                <p>Your AI Trading Agent email alerts are configured correctly.</p>
                <p style="color:#888;font-size:13px">Sent at: {now}</p>
              </div>
            </body></html>
            """
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = self.from_addr
            msg["To"]      = self.to_addr
            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.from_addr, self.password)
                server.sendmail(self.from_addr, self.to_addr, msg.as_string())

            return True, f"Test email sent to {self.to_addr}"
        except Exception as e:
            return False, f"Failed: {e}"

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format(self, fire) -> tuple[str, dict]:
        # Handle both dataclass and dict
        if hasattr(fire, "__dict__"):
            f = fire.__dict__
        elif hasattr(fire, "__dataclass_fields__"):
            from dataclasses import asdict
            f = asdict(fire)
        else:
            f = fire

        atype   = f.get("alert_type", "ALERT")
        symbol  = f.get("symbol", "")
        message = f.get("message", "Alert triggered")
        val     = f.get("value", 0)
        thr     = f.get("threshold", 0)
        fired   = f.get("fired_at", "")[:16].replace("T", " ")

        icon_map = {
            "PRICE_ABOVE":    "📈",
            "PRICE_BELOW":    "📉",
            "RSI_OVERBOUGHT": "🔴",
            "RSI_OVERSOLD":   "🟢",
            "SIGNAL_CHANGE":  "🎯",
            "NEWS_SENTIMENT": "📰",
        }
        icon    = icon_map.get(atype, "🔔")
        subject = f"{icon} Trading Alert: {symbol} — {atype.replace('_', ' ').title()}"

        color_map = {
            "PRICE_ABOVE":    "#00cc66",
            "PRICE_BELOW":    "#ff4444",
            "RSI_OVERBOUGHT": "#ff8800",
            "RSI_OVERSOLD":   "#00cc66",
            "SIGNAL_CHANGE":  "#00aaff",
            "NEWS_SENTIMENT": "#aa88ff",
        }
        color = color_map.get(atype, "#888888")

        text = f"{icon} {subject}\n\n{message}\n\nValue: {val}\nThreshold: {thr}\nTime: {fired} UTC\n\n— AI Trading Agent"

        html = f"""
        <html><body style="font-family:sans-serif;background:#0f0f1a;color:#eee;padding:20px;margin:0">
          <div style="max-width:520px;margin:0 auto">
            <div style="background:#1a1a2e;border-radius:10px;overflow:hidden">
              <div style="background:{color};padding:16px 24px">
                <h2 style="margin:0;color:white;font-size:20px">{icon} {atype.replace("_", " ").title()}</h2>
                <p style="margin:4px 0 0;color:rgba(255,255,255,0.85);font-size:14px">{symbol}</p>
              </div>
              <div style="padding:20px 24px">
                <p style="font-size:16px;color:#eee;margin:0 0 16px">{message}</p>
                <table style="width:100%;border-collapse:collapse">
                  <tr>
                    <td style="padding:8px 0;color:#888;font-size:13px">Triggered value</td>
                    <td style="padding:8px 0;color:{color};font-weight:bold;text-align:right">{val}</td>
                  </tr>
                  <tr>
                    <td style="padding:8px 0;color:#888;font-size:13px;border-top:1px solid #333">Threshold</td>
                    <td style="padding:8px 0;color:#ccc;text-align:right;border-top:1px solid #333">{thr}</td>
                  </tr>
                  <tr>
                    <td style="padding:8px 0;color:#888;font-size:13px;border-top:1px solid #333">Time (UTC)</td>
                    <td style="padding:8px 0;color:#ccc;text-align:right;border-top:1px solid #333">{fired}</td>
                  </tr>
                </table>
              </div>
              <div style="padding:12px 24px;background:#111;border-top:1px solid #222">
                <p style="margin:0;color:#555;font-size:12px">AI Trading Agent — Paper Trading System</p>
              </div>
            </div>
          </div>
        </body></html>
        """

        return subject, {"text": text, "html": html}


def make_email_sender_from_settings(settings) -> "EmailSender":
    """Factory that reads from Pydantic settings — returns EmailSender or None."""
    try:
        from_addr = getattr(settings, "ALERT_EMAIL_FROM",     None) or ""
        password  = getattr(settings, "ALERT_EMAIL_PASSWORD", None) or ""
        to_addr   = getattr(settings, "ALERT_EMAIL_TO",       None) or from_addr
        host      = getattr(settings, "ALERT_SMTP_HOST",      "smtp.gmail.com")
        port      = int(getattr(settings, "ALERT_SMTP_PORT",  587))
        return EmailSender(from_addr, password, to_addr, host, port)
    except Exception as e:
        logger.warning(f"EmailSender init failed: {e}")
        return EmailSender("", "", "")
