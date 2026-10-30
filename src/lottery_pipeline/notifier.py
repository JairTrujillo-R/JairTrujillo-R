from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage


@dataclass(frozen=True)
class SMTPSettings:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipient: str
    use_tls: bool = True

    @staticmethod
    def from_env() -> SMTPSettings:
        host = os.getenv("LOTTERY_SMTP_HOST", "")
        port = int(os.getenv("LOTTERY_SMTP_PORT", "587"))
        username = os.getenv("LOTTERY_SMTP_USERNAME", "")
        secret_value = os.getenv("LOTTERY_SMTP_PASSWORD", "")
        sender = os.getenv("LOTTERY_SMTP_SENDER", "")
        recipient = os.getenv("LOTTERY_ALERT_RECIPIENT", "")
        use_tls = os.getenv("LOTTERY_SMTP_USE_TLS", "true").lower() == "true"
        return SMTPSettings(host, port, username, secret_value, sender, recipient, use_tls)

    def is_complete(self) -> bool:
        return all(
            [self.host, self.port, self.username, self.password, self.sender, self.recipient]
        )


class EmailNotifier:
    def __init__(self, settings: SMTPSettings) -> None:
        self.settings = settings

    def send_hit_alert(self, subject: str, body: str) -> str:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.settings.sender
        message["To"] = self.settings.recipient
        message.set_content(body)

        with smtplib.SMTP(self.settings.host, self.settings.port, timeout=30) as smtp:
            if self.settings.use_tls:
                smtp.starttls()
            smtp.login(self.settings.username, self.settings.password)
            smtp.send_message(message)
        return "sent"
