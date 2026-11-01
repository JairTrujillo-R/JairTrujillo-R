from __future__ import annotations

import json
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from html import escape
from typing import Any


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
        port = int(os.getenv("LOTTERY_SMTP_PORT") or "587")
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

    def send_hit_alert(self, subject: str, body: str, html_body: str | None = None) -> str:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.settings.sender
        message["To"] = self.settings.recipient
        message.set_content(body)
        if html_body:
            message.add_alternative(html_body, subtype="html")

        with smtplib.SMTP(self.settings.host, self.settings.port, timeout=30) as smtp:
            if self.settings.use_tls:
                smtp.starttls()
            smtp.login(self.settings.username, self.settings.password)
            smtp.send_message(message)
        return "sent"


def render_hit_email(rows: list[Any]) -> tuple[str, str]:
    """Render one accessible plain-text and HTML report for one or more hits."""
    count = len(rows)
    title = "Acierto de lotería" if count == 1 else f"{count} aciertos de lotería"
    plain_lines = [title, "", "Detalle:"]
    html_rows: list[str] = []
    major_hits = 0
    secondary_hits = 0
    for row in rows:
        matched = row["matched_prize_types"] if "matched_prize_types" in row.keys() else []
        if isinstance(matched, str):
            try:
                matched = json.loads(matched)
            except json.JSONDecodeError:
                matched = [matched]
        matched_text = ", ".join(matched) if matched else "premio"
        major_hits += int("mayor" in matched)
        secondary_hits += int(any(item != "mayor" for item in matched))
        plain_lines.append(
            f"- {row['prediction_date']} | {row['strategy']} / {row['game']} | "
            f"Predicho: {row['predicted_number']} | Coincide: {matched_text}"
        )
        html_rows.append(
            "<tr>"
            f"<td>{escape(str(row['prediction_date']))}</td>"
            f"<td>{escape(str(row['strategy']))}</td>"
            f"<td>{escape(str(row['game']))}</td>"
            f"<td><strong>{escape(str(row['predicted_number']))}</strong></td>"
            f"<td>{escape(matched_text)}</td>"
            "</tr>"
        )
    plain_lines.extend(
        ["", f"Premios mayores: {major_hits}", f"Premios secundarios: {secondary_hits}"]
    )
    html = f"""<!doctype html>
<html lang="es"><body style="margin:0;background:#f4f7fb;color:#172033;
font-family:Arial,sans-serif">
<div style="max-width:760px;margin:24px auto;background:#fff;border:1px solid #dfe6ef;
border-radius:12px;overflow:hidden">
  <div style="background:#123b63;color:#fff;padding:24px 28px">
    <div style="font-size:12px;letter-spacing:1px;text-transform:uppercase;
opacity:.8">Lottery Pipeline</div>
    <h1 style="margin:8px 0 0;font-size:26px">{escape(title)}</h1>
  </div>
  <div style="padding:24px 28px">
    <p style="margin-top:0">Se detectaron coincidencias exactas con resultados oficiales.</p>
    <div style="display:flex;gap:12px;margin:20px 0">
      <div style="background:#e8f5ee;padding:12px 16px;border-radius:8px">
        <b>{count}</b><br><small>Aciertos</small>
      </div>
      <div style="background:#eef3f8;padding:12px 16px;border-radius:8px">
        <b>{major_hits}</b><br><small>Mayor</small>
      </div>
      <div style="background:#fff4df;padding:12px 16px;border-radius:8px">
        <b>{secondary_hits}</b><br><small>Secundarios</small>
      </div>
    </div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="background:#f0f3f7;text-align:left">
          <th style="padding:11px">Fecha</th><th style="padding:11px">Estrategia</th>
          <th style="padding:11px">Juego</th><th style="padding:11px">Número</th>
          <th style="padding:11px">Premio</th>
        </tr></thead>
        <tbody>{"".join(html_rows)}</tbody>
      </table>
    </div>
    <p style="color:#68758a;font-size:12px;margin-bottom:0">
      El resultado es una coincidencia histórica y no garantiza resultados futuros.
    </p>
  </div>
</div></body></html>"""
    return "\n".join(plain_lines), html
