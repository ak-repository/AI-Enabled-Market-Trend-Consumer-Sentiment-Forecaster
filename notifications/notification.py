import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def send_mail(text, subject, df=None):

    try:
        sender = os.getenv("sender_gmail")
        password = os.getenv("Google_app_password")
        receiver = os.getenv("reciver_gmail")

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = subject

        # Email body
        msg.attach(MIMEText(text, "plain"))

        # ✅ Excel attachment
        if df is not None and not df.empty:
            filename = f"sentiment_alert_{datetime.now().date()}.xlsx"

            # Excel file create
            df.to_excel(filename, index=False)

            with open(filename, "rb") as f:
                part = MIMEBase(
                    "application",
                    "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                part.set_payload(f.read())

            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{filename}"'
            )

            msg.attach(part)
            os.remove(filename)

        # Send mail
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        print("Email sent Successfully")
        return True

    except Exception as e:
        print("Email failed:", e)
        return False







# import smtplib
# from email.mime.text import MIMEText
# import os
# from dotenv import load_dotenv
# import requests

# load_dotenv()

# def send_mail(text, subject, df=None ):
    
    
#     try:
#         # email details
#         sender = os.getenv("sender_gmail")
#         password = os.getenv("Google_app_password")
#         reciver = os.getenv("reciver_gmail")

#         # create message 
#         msg = MIMEText(text)
#         msg["Subject"] = subject
#         msg["From"] = sender
#         msg["To"]= reciver
        
#         # connection to gamil SMTP server
#         with smtplib.SMTP("smtp.gmail.com", 587) as server:
#             server.starttls()  #start TLS encryption
#             server.login(sender, password)
#             server.send_message(msg)
            
#         print("Email sent Successfully")
#         return True
    
#     except Exception as e:
#         print("Email failed:", e)
#         return False


# Slack

def send_slack_notifications(text):

    webhooks_url=os.getenv("Incoming_webhooks")

    message= {
        "text": text,
        "username": "AI-Enabled Market Trend & Consumer Sentiment Forecaster",
        "icon_emoji": ":shield"
    }

    requests.post(webhooks_url, json=message)


def testing_function():
    print("Hello World!")
