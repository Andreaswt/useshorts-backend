import resend


def send_clip_generated_email(email, resend_api_key, bucketUrl):
    print("Sending generated email")
    resend.api_key = resend_api_key
    params = {
        "from": "Video Scheduler <help@useshorts.app>",
        "to": [email],
        "subject": "Your clip is ready",
        "html": f"""
        <html>
        <body style="background-color: #ffffff; font-family: Arial, sans-serif; margin: 0; padding: 0;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #333; font-size: 24px; font-weight: bold; margin: 40px 0;">Your clip is ready</h1>
                <p style="color: #333; font-size: 14px; line-height: 1.5;">
                    Download it below, or check it out in the dashboard.
                </p>
                <a href="{bucketUrl}" style="background-color: #918CF2; color: #ffffff; display: inline-block; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 20px 0; font-weight: 600;">View Scheduled Video</a>
                <p style="color: #333; font-size: 14px; line-height: 1.5;">Cheers,<br>UseShorts</p>
                <p style="color: #898989; font-size: 12px; line-height: 1.5; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee;">
                    PS: You can get 16 free videos by referring a friend to UseShorts, so don't hesitate to share <a href='https://app.useshorts.app/dashboard/refer' style="color: #2754C5; text-decoration: underline;">your referral link</a> with your friends!
                </p>
            </div>
        </body>
        </html>
        """
    }
    resend.Emails.send(params)
