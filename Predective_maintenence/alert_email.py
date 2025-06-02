# alert_notifier.py

import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SENT_LOG_FILE = "notified_techs.log"
STATIC_PDF_PATH = "Volltext-PDF.pdf"

class ScheduleAlertTool:
    def __init__(self, schedule_path: str, tech_path: str):
        self.schedule_df = pd.read_csv(schedule_path, parse_dates=["scheduled_start", "scheduled_end"])
        self.technicians_df = pd.read_csv(tech_path)
        self.notified_techs = self._load_notified_techs()

    def _load_notified_techs(self):
        if not os.path.exists(SENT_LOG_FILE):
            return set()
        with open(SENT_LOG_FILE, "r") as file:
            return set(line.strip() for line in file.readlines())

    def _mark_as_notified(self, tech_id):
        with open(SENT_LOG_FILE, "a") as file:
            file.write(f"{tech_id}\n")

    def send_email(self, recipient_email, subject, body, pdf_path: Optional[str] = None):
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_path)}"'
            msg.attach(part)

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

    def notify_one(self) -> str:
        if self.schedule_df.empty:
            return "\u26a0\ufe0f No schedule entries found."

        self.schedule_df = self.schedule_df.sort_values("scheduled_start")

        for _, row in self.schedule_df.iterrows():
            tech_id = str(row['technician_id'])
            if tech_id in self.notified_techs:
                continue

            tech_info = self.technicians_df[self.technicians_df['technician_id'] == int(tech_id)]
            if tech_info.empty:
                return f"❌ Technician with ID {tech_id} not found."

            email = tech_info.iloc[0]['gmail']
            name = tech_info.iloc[0]['name']

            maintenance_date = row['scheduled_start'].strftime('%Y-%m-%d')
            subject = f"🛠 Maintenance Scheduled for {row['machine_id']} on {maintenance_date}"
            body = (
                f"Hello {name},\n\n"
                f"You have been scheduled to attend the anomaly on Machine ID: {row['machine_id']} on {row['scheduled_start']}\n"
                f"Scheduled Start: {row['scheduled_start']}\n"
                f"Scheduled End: {row['scheduled_end']}\n\n"
                "Please make the necessary arrangements.\n\nRegards,\nScheduler Bot"
            )

            self.send_email(email, subject, body, pdf_path=STATIC_PDF_PATH)
            self._mark_as_notified(tech_id)
            return f"✅ Notification sent to {name} ({email}) for machine {row['machine_id']}."

        return "🔁 No new technicians to notify. All scheduled technicians have already been notified."

def alert_notification(_: str = "") -> str:
    alert_tool = ScheduleAlertTool("schedule.csv", "Tech_availability.csv")
    return alert_tool.notify_one()

def Alert_agent():
    tools = [
        Tool(
            name="SendScheduleAlerts",
            func=alert_notification,
            description="Sends one email notification to the first unnotified technician in schedule.csv"
        )
    ]
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

if __name__ == "__main__":
    agent = Alert_agent()
    print("📬 Sending Alert Email...")
    result = agent.run("send email alert to technician")
    print(result)
