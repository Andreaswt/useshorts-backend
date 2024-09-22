import resend


def send_invalid_yt_oauth_email(email, resend_api_key):
    print("Sending invalid YT oauth email")
    resend.api_key = resend_api_key
    params = {
        "from": "Shorts <errors@useshorts.app>",
        "to": [email],
        "reply_to": "help@useshorts.app",
        "subject": "YouTube login no longer valid",
        "html": f"""
        <html>
        <body style="background-color: #ffffff; font-family: Arial, sans-serif; margin: 0; padding: 0;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #333; font-size: 24px; font-weight: bold; margin: 40px 0;">YouTube login no longer valid</h1>
                <p style="color: #333; font-size: 14px; line-height: 1.5;">
                    Your YouTube login is no longer valid, or doesn't have sufficient permissions. Please go to app.useshorts.app and log back in with your YouTube channel again to use autoposting.
                </p>
                <a href="https://app.useshorts.app" style="background-color: #918CF2; color: #ffffff; display: inline-block; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 20px 0; font-weight: 600;">Log in to UseShorts</a>
                <p style="color: #333; font-size: 14px; line-height: 1.5;">Best,<br>UseShorts</p>
                <p style="color: #898989; font-size: 12px; line-height: 1.5; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee;">
                    If you need any assistance, please don't hesitate to reply to this email or contact me at help@useshorts.app.
                </p>
            </div>
        </body>
        </html>
        """
    }
    resend.Emails.send(params)
