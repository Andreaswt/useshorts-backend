import resend


def send_timeout_email(context, resend_api_key, clip_id):
    print("Sending timeout email")
    if context is not None:
        region = context.invoked_function_arn.split(":")[3]
        log_group_name = context.log_group_name
        log_stream_name = context.log_stream_name
        log_url = f"""https://console.aws.amazon.com/cloudwatch/home?region={
            region}#logEventViewer:group={log_group_name};stream={log_stream_name}"""

    resend.api_key = resend_api_key
    params = {
        "from": "Error detector <errors@useshorts.app>",
        "to": ["errors@useshorts.app"],
        "subject": "Error in shorts-clipper (2nd lambda)",
        "html": f"<div>Timeout in 2nd lambda (clip_id={str(clip_id)})<br/>Cloudwatch url: {log_url}</div>"
    }
    resend.Emails.send(params)
