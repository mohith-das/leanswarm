from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

from leanswarm.webui.config import WebUISettings


def send_password_reset_email(to_email: str, reset_url: str, settings: WebUISettings) -> None:
    msg = MIMEText(
        f"Someone requested a password reset for your account.\n\n"
        f"Reset your password: {reset_url}\n\n"
        f"This link expires in 1 hour. If you did not request this, you can ignore this email.",
        "plain",
    )
    msg["From"] = settings.from_email
    msg["To"] = to_email
    msg["Subject"] = "Password reset"

    with smtplib.SMTP_SSL(settings.smtp_host or "", settings.smtp_port) as server:
        server.login(settings.smtp_username or "", settings.smtp_password or "")
        server.sendmail(settings.from_email, to_email, msg.as_string())
