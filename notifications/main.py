from notification import send_slack_notifications, send_mail

send_mail(text="Hello this is test user", subject="Testing mail")

send_slack_notifications(text="Hello this is my first notification")