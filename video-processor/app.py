import json
import os
import random
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from datetime import timedelta

import boto3
import isodate
import resend
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from dotenv import load_dotenv
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from prisma.enums import SchedulingOrder
from pytubefix import YouTube

from emails.invalid_yt_oauth import send_invalid_yt_oauth_email
from emails.playlist_missing import send_playlist_missing_email
from prisma import Prisma
import cuid

from emails.timeout import send_timeout_email

load_dotenv()


def analyze_transcript_deepgram(input_file_path, deepgram_audio_file_path):
    print("Analyze_transcript_deepgram is called")

    deepgram = DeepgramClient()

    # Sending audio to Deepgram is faster
    print("Extracting audio")
    time_start = time.time()
    command = [
        'ffmpeg', '-i', input_file_path, '-ar',
        '16000', '-ac', '1', '-loglevel', 'error', '-hide_banner', deepgram_audio_file_path
    ]
    subprocess.call(command)
    time_end = time.time()
    print(f"""Audio extracted in {time_end - time_start}""")

    print("Opening file")
    with open(deepgram_audio_file_path, "rb") as file:
        buffer_data = file.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    options = PrerecordedOptions(
        model="nova-2-video",
        smart_format=True,
        paragraphs=True,
        detect_language=True,
        utterances=True,
    )

    print("Transcribing with Deepgram")
    time_start = time.time()
    deepgram_response = deepgram.listen.prerecorded.v(
        "1").transcribe_file(payload, options, timeout=300)
    time_end = time.time()
    print(f"""Deepgram transcription done in {time_end - time_start}""")

    return deepgram_response


def identify_questions_and_answers(bedrock, sentences):
    ### Questions and Answers ###
    qna_body = json.dumps({
        "max_tokens": 180000,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": """
This is a podcast video transcript consisting of sentences, along with each sentence's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

Your task is to find and extract every single question and its corresponding answer from the transcript without exceptions. Each clip should begin with the question and conclude with the answer. It is acceptable for the clip to include a few additional sentences before the question if it aids in contextualizing the question.

Please adhere to the following rules:
- Ensure that clips do not overlap with one another.
- Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
- Only use the start and end timestamps provided in the input; modifying timestamps is not allowed.
- Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
- Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

Avoid including:
- Moments of greeting, thanking, or saying goodbye.
- Non-question and answer interactions.

If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

The transcript is as follows:\n\n""" + str(sentences)
            },
            {
                "role": "assistant",
                "content": "Here are the curated podcast highlights in JSON format:"
            }
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }
    )

    max_retries = 3
    for i in range(max_retries):
        try:
            print("Prompting qna")
            prompt_start = time.time()
            qna_response = bedrock.invoke_model(
                body=qna_body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0")
            prompt_end = time.time()
            print(f"""Prompting qna took {
                prompt_end - prompt_start} seconds""")

            break

        except Exception as e:
            if i < max_retries - 1:
                time.sleep(5)  # wait a bit before trying again
                print(f"""Retrying questions prompt... Attempt {
                      i+1} failed with error: {e}""")
                continue
            else:
                raise
    try:
        body = qna_response.get("body").read()
        json_body = json.loads(body)
        body_text = json_body["content"][0]["text"]
        print(str(body_text))

        return json.loads(body_text)
    except Exception as e:
        print(
            "Error in json decoding identify_questions_and_answers. Returning []: " + str(e))
        return []


def identify_statements(bedrock, sentences, qnas):
    ### Statements ###
    statements_body = json.dumps({
        "max_tokens": 180000,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": """
This is a podcast video transcript along with each sentence's start and end time. I aim to create video clips lasting between a minimum of 30 and a maximum of 60 seconds. These clips should never exceed 60 seconds.

Your task is to find and extract all full stories, detailed explanations of topics, or expressed opinions. Each clip should start at the beginning of these segments and include as much of the content as possible without exceeding 60 seconds.

Please adhere to the following guidelines:
- Ensure that clips do not overlap with each other or with periods previously identified as Q&A sessions.
- Clips must not start or end in the middle of a sentence. You must use only the exact start and end timestamps provided in the transcript.
- Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
- Keep all clips under 60 seconds.
- Exclude any non-content moments such as thank yous, greetings, or goodbyes.
- Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.
- Ensure there is a significant gap between clips to avoid redundancy and repetitive content.

If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

Note: Below is a list of moments previously identified as containing questions and answers. Make sure your selected clips do not overlap with these intervals.

"""
                + str(qnas) +

                """
Here is the transcript:

"""
                + str(sentences)
            },
            {
                "role": "assistant",
                "content": "Here is the JSON:"
            }
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }
    )

    max_retries = 3
    for i in range(max_retries):
        try:
            print("Prompting statements")
            prompt_start = time.time()
            statements_response = bedrock.invoke_model(
                body=statements_body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0")
            prompt_end = time.time()
            print(f"""Prompting statements took {
                prompt_end - prompt_start} seconds""")
            break

        except Exception as e:
            if i < max_retries - 1:
                time.sleep(5)  # wait a bit before trying again
                print(f"""Retrying questions prompt... Attempt {
                      i+1} failed with error: {e}""")
                continue
            else:
                raise

    try:
        body = statements_response.get("body").read()
        json_body = json.loads(body)
        body_text = json_body["content"][0]["text"]
        print(str(body_text))

        return json.loads(body_text)
    except Exception as e:
        print("Error in json decoding identify_statements. Returning []: " + str(e))
        print("Response that caused error: " +
              str(statements_response.get("body").read()))
        return []


def identify_moments(deepgram_response):
    paragraphs = deepgram_response.get("results", {}).get("channels", [{}])[0].get(
        "alternatives", [{}])[0].get("paragraphs", []).get("paragraphs", [])

    all_sentences = []

    for paragraph in paragraphs:
        sentences = paragraph.get("sentences", [])
        all_sentences.extend(sentences)

    bedrock = boto3.client(service_name="bedrock-runtime",
                           region_name="us-east-1")

    def is_overlapping(new_moment, existing_moments):
        """
        Check if new_moment overlaps with any moment in existing_moments.
        Each moment is assumed to have 'start' and 'end' properties.
        """
        new_start, new_end = new_moment['start'], new_moment['end']
        for moment in existing_moments:
            # Check for overlap
            if not (new_end <= moment['start'] or new_start >= moment['end']):
                return True
        return False

    def is_valid_duration(clip):
        duration = clip['end'] - clip['start']
        return 30 < duration < 60

    def extend_moments(bedrock, all_sentences, existing_moments, identify_function, overlap_check_moments=None):
        def is_valid_moment(moment):
            return is_valid_duration(moment) and not is_overlapping(moment, existing_moments) and (overlap_check_moments is None or not is_overlapping(moment, overlap_check_moments))

        new_potential_moments = identify_function(bedrock, all_sentences)
        filtered_moments = [
            moment for moment in new_potential_moments if is_valid_moment(moment)]

        # If there are fewer than 3 valid results, try one more time and extend the list
        if len(filtered_moments) < 3:
            additional_moments = identify_function(bedrock, all_sentences)
            filtered_moments.extend(
                [moment for moment in additional_moments if is_valid_moment(moment)])

        for moment in filtered_moments:
            if is_valid_moment(moment):
                existing_moments.append(moment)

        return existing_moments

    # Identify QNA moments
    qna_moments = []
    qna_moments = extend_moments(
        bedrock, all_sentences, qna_moments, identify_questions_and_answers)

    # Identify statement moments
    def identify_statements_wrapper(b, s):
        return identify_statements(b, s, qna_moments)

    statements_moments = []
    statements_moments = extend_moments(bedrock, all_sentences, statements_moments,
                                        identify_statements_wrapper,
                                        overlap_check_moments=qna_moments)

    print("Transcripts for Q&A clips:\n" + json.dumps(qna_moments, indent=4))
    print("Transcripts for statements clips:\n" +
          json.dumps(statements_moments, indent=4))

    qna_moments.extend(statements_moments)

    return qna_moments


def download_video(url, filename, to_path):
    last_exception = None
    for attempt in range(3):
        try:
            yt = YouTube(url)
            video_streams = sorted(
                [stream for stream in yt.streams.filter(file_extension='mp4')
                    if stream.resolution and int(stream.resolution.replace('p', '')) <= 1080],
                key=lambda s: int(s.resolution.replace('p', '')), reverse=True)

            if video_streams[0].is_progressive:
                output_path = os.path.join(to_path, filename)
                video_streams[0].download(
                    output_path=to_path, filename='temp_video.mp4')

                command = (
                    f"ffmpeg - i {
                        os.path.join(to_path, 'temp_video.mp4')} "
                    f"- c: v copy - c: a copy - loglevel error - hide_banner {
                        output_path}"
                )
                subprocess.call(command, shell=True)
                # Clean up temp file
                os.remove(os.path.join(to_path, 'temp_video.mp4'))
            else:
                video_stream = video_streams[0]
                audio_stream = yt.streams.get_audio_only()

                video_stream.download(output_path=to_path,
                                      filename='temp_video.mp4')
                audio_stream.download(output_path=to_path,
                                      filename='temp_audio.mp4')

                video_path = os.path.join(to_path, 'temp_video.mp4')
                audio_path = os.path.join(to_path, 'temp_audio.mp4')
                output_path = os.path.join(to_path, filename)

                command = (
                    f"ffmpeg -i {video_path} "
                    f"-i {audio_path} "
                    f"- c: v copy - c: a aac - loglevel error - hide_banner {
                        output_path}"
                )
                subprocess.call(command, shell=True)
                os.remove(video_path)
                os.remove(audio_path)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            last_exception = e
            time.sleep(15)  # Wait before retrying
    else:
        print("Failed to download video after 3 attempts")
        if last_exception is not None:
            raise last_exception


def video_available(youtube_video_id):
    try:
        yt = YouTube('https://www.youtube.com/watch?v=' + youtube_video_id)
        yt.check_availability()
    except:
        return False

    return True


def get_channel_video(db: Prisma, user_id, scheduling_order: SchedulingOrder, user_email, userDefinedVideoId, is_autoposting):
    print("Building public YouTube client")
    api_key = os.getenv('YOUTUBE_API_KEY')
    public_youtube_client = build('youtube', 'v3', developerKey=api_key)

    if is_autoposting:
        print("Getting OAuth token")
        oauthToken = db.oauth.find_unique(
            where={
                "provider_userId": {
                    "userId": user_id,
                    "provider": "youtube"
                }
            }
        )

        if oauthToken.invalid == True:
            print("Invalid refresh token for user with id: " + user_id)
            return None, None, None, None

        if oauthToken.selectedChannelId == None:
            print("No channel selected for user with id: " +
                  user_id + ". Lambda shouldn't be called.")
            return None, None, None, None

        refresh_token = oauthToken.refreshToken
        channel_id = oauthToken.selectedChannelId

        client_id = os.getenv('YOUTUBE_CLIENT_ID')
        client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')

        credentials = Credentials.from_authorized_user_info({
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret
        })

        print("Building oauth YouTube client")
        oauthYoutube = build('youtube', 'v3', credentials=credentials)

        print("Getting videos from playlist or channel")
        video_ids = []

        if oauthToken.selectedPlaylistId:
            playlist_check = oauthYoutube.playlists().list(
                part="snippet",
                id=oauthToken.selectedPlaylistId
            ).execute()

            if playlist_check['items']:  # Playlist exists
                page_token = None
                while True:
                    print(f"Fetching videos from playlist: {
                          oauthToken.selectedPlaylistId}")
                    playlist_response = oauthYoutube.playlistItems().list(
                        part="snippet",
                        playlistId=oauthToken.selectedPlaylistId,
                        maxResults=50,
                        pageToken=page_token
                    ).execute()
                    # Collect all video IDs from the playlist response
                    video_ids.extend([item['snippet']['resourceId']['videoId']
                                      for item in playlist_response['items']])

                    page_token = playlist_response.get('nextPageToken')
                    if not page_token:
                        break
            else:
                print(
                    "Playlist doesn't exist. Setting selectedPlaylistId to None, and stopping lambda.")
                db.oauth.update(
                    where={
                        "provider_userId": {
                            "userId": user_id,
                            "provider": "youtube"
                        }
                    },
                    data={
                        "selectedPlaylistId": None,
                        "selectedPlaylistName": None,
                        "selectedPlaylistImage": None
                    }
                )
                send_playlist_missing_email(
                    user_email, os.environ["RESEND_API_KEY"])
                return None, None, None, None
        else:
            print(f"Fetching videos from channel: {channel_id}")
            page_token = None
            while True:
                video_response = oauthYoutube.search().list(
                    part="snippet",
                    channelId=channel_id,
                    maxResults=50,
                    type="video",
                    order="date",
                    pageToken=page_token
                ).execute()

                # Collecting all video IDs from the search response
                video_ids.extend([
                    item['id']['videoId']
                    for item in video_response['items']
                    if item['snippet']['liveBroadcastContent'] == 'none'
                ])

                page_token = video_response.get('nextPageToken')
                if not page_token:
                    break
    else:  # Single generated video
        video_ids = [userDefinedVideoId]

    if os.environ["ENVIRONMENT"] == "local":
        video_ids = ['AxuwazaXOMg']

    all_users_videos = db.video.find_many(
        where={
            "userId": user_id,
            "singleGeneratedVideo": not is_autoposting
        }
    )

    # Get all processed video ids
    all_users_processed_video_ids = [
        video.youtube_id for video in all_users_videos if video.isProcessed == True]

    # Make sure we only get videos that haven't been processed yet
    video_ids = [
        video_id for video_id in video_ids if video_id not in all_users_processed_video_ids]

    # public_youtube_client.videos().list can only take 50 at a time. takes 50 unprocessed videos.
    if len(video_ids) > 0:
        if scheduling_order == "SHUFFLE":
            video_ids = random.sample(video_ids, min(len(video_ids), 50))
        elif scheduling_order == "CHRONOLOGICAL":
            video_ids = video_ids[:50]

    # Fetching details for all videos at once
    videos_response = public_youtube_client.videos().list(
        part='contentDetails,snippet,status',
        id=','.join(video_ids)
    ).execute()

    if scheduling_order == "SHUFFLE":
        # Shuffle videos to avoid a lot of clips being posted in a row from the same video
        random.shuffle(videos_response['items'])

    for item in videos_response['items']:
        if not video_available(item['id']) or item['status']['uploadStatus'] != 'processed' or item['status']['privacyStatus'] != 'public':
            print(
                "Video is not available, not processed or it's private. Skipping video.")
            continue

        video_id = item['id']
        duration = isodate.parse_duration(
            item['contentDetails']['duration'])

        title = item['snippet']['title']
        description = item['snippet']['description']

        if timedelta(minutes=5) < duration < timedelta(minutes=90):
            existing_db_video = next(
                (x for x in all_users_videos if x.youtube_id == video_id), None
            )

            if existing_db_video is not None:
                if existing_db_video.isProcessed:
                    print("Existing DB video is already processed")
                    continue

                print("Existing DB video id is " + existing_db_video.id)
                return existing_db_video.id, video_id, title, description
            else:
                new_db_video = db.video.create(
                    data={
                        "youtube_id": video_id,
                        "userId": user_id,
                        "singleGeneratedVideo": not is_autoposting,
                    }
                )
                print("Created DB video with ID " + new_db_video.id)
                return new_db_video.id, video_id, title, description

    return None, None, None, None


def set_status_for_queued_video(db: Prisma, queuedVideoId, status):
    db.queuedvideo.update(
        where={
            "id": queuedVideoId
        },
        data={
            "status": status
        }
    )


def set_failed_status_for_queued_video(db: Prisma, user_id, queuedVideoId):
    set_status_for_queued_video(db, queuedVideoId, "Failed")

    db.user.update(
        where={
            "id": user_id
        },
        data={
            "credits": {
                "increment": 1
            }
        }
    )


def create_and_upload_clip(unique_dirname, input_file_path, start, end, bucket_name, folder_name, video_id, user_id):
    print("Creating and uploading clip from " + str(start) + " to " + str(end))
    clip_id = cuid.cuid()

    # Use the clip_id for the filename
    clip_filename = f"{clip_id}.mp4"
    clip_path = os.path.join(unique_dirname, clip_filename)

    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "error",
        "-hide_banner",
        "-ss", str(start),
        "-i", input_file_path,
        "-t", str(end - start),
        "-c", "copy",
        clip_path
    ]

    subprocess.run(ffmpeg_command, check=True)

    print("Uploading clip to S3")
    s3_client = boto3.client('s3')
    s3_key = f"{folder_name}/{clip_filename}"
    s3_client.upload_file(clip_path, bucket_name, s3_key)

    bucket_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

    return {
        "id": clip_id,
        "userId": user_id,
        "videoId": video_id,
        "rawClipBucketUrl": bucket_url,
        "startTimeInOriginalVideoMs": int(start * 1000),
        "endTimeInOriginalVideoMs": int(end * 1000),
    }


def return_success():
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "success"
        })
    }


def return_failure(messageId):
    return {
        "statusCode": 200,
        "body": json.dumps({
            "batchItemFailures": [
                {
                    "ItemIdentifier": messageId,
                    "SenderFault": True,
                }
            ]
        })
    }


def main(user_id, userDefinedVideoId=None, queuedVideoId=None):
    is_autoposting = userDefinedVideoId is None and queuedVideoId is None

    try:
        if os.environ["ENVIRONMENT"] == "local":
            temp_dir = "./tmp"
        else:
            temp_dir = "/tmp"

        unique_dirname = os.path.join(
            temp_dir, f"{uuid.uuid4()}".replace('-', ''))
        print("unique_dirname is " + unique_dirname)

        print("Connecting to db")
        db = Prisma()
        db.connect()
        print("Connected to db")

        if not is_autoposting:
            set_status_for_queued_video(db, queuedVideoId, "Processing 10%")

        print("DB: Looking for user with id " + user_id)
        user = db.user.find_unique_or_raise(
            where={
                "id": user_id
            }
        )

        if os.environ["ENVIRONMENT"] == "production" and user.isRunning == False and is_autoposting:
            print("UseShorts not running for user with id: " + user.id)
            return

        if os.environ["ENVIRONMENT"] == "production" and is_autoposting:
            unused_clips = db.clip.find_many(
                where={
                    "userId": user_id,
                    "bucketUrl": None
                }
            )

            if user.schedulingOrder == SchedulingOrder.CHRONOLOGICAL:
                unused_clips = sorted(
                    unused_clips, key=lambda x: x.createdAt, reverse=False)
            elif user.schedulingOrder == SchedulingOrder.SHUFFLE:
                random.shuffle(unused_clips)

            if len(unused_clips) > 0:
                print("Unused clips found for user with id: " +
                      user.id + ". Sending message to the next SQS queue.")
                sqs = boto3.client('sqs', region_name=os.environ["AWS_REGION"])

                clip_id = next(iter(unused_clips)).id
                message_body = json.dumps({
                    'clipId': clip_id,
                    **({'queuedVideoId': queuedVideoId} if queuedVideoId is not None else {})
                })

                print("Clip id: " + str(clip_id))

                # Send the message
                sqs.send_message(
                    QueueUrl=os.environ["SHORTS_CLIPPER_QUEUE_URL"], MessageBody=message_body)
                return

        print("Getting channel video")
        get_channel_video_start = time.time()
        db_video_id, youtube_video_id, long_videos_title, long_videos_description = get_channel_video(
            db, user_id, user.schedulingOrder, user.email, userDefinedVideoId, is_autoposting)
        get_channel_video_end = time.time()
        print(f"""Getting channel video took {
              get_channel_video_end - get_channel_video_start} seconds""")

        if (db_video_id is None or youtube_video_id is None):
            if not is_autoposting:
                set_failed_status_for_queued_video(db, user_id, queuedVideoId)
            else:
                print(
                    "All videos have been uploaded from channel, for user with id: " + user.id)
            return

        print("YouTube video id: " + str(youtube_video_id))

        print("Creating s3 client")
        s3_resource = boto3.resource('s3')
        bucket_name = os.environ["SHORTS_PROCESSED_VIDEOS"]
        folder_name = str(db_video_id)

        objects = s3_resource.Bucket(
            bucket_name).objects.filter(Prefix=folder_name)
        folder_exists = any(True for _ in objects)

        if folder_exists:
            print("Video already exists in S3. Skipping. Folder name: " + folder_name)
        else:
            print("Video doesn't exist in S3")
            print("Creating tmp and unique dirs")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(unique_dirname, exist_ok=True)

            url = 'https://www.youtube.com/watch?v=' + youtube_video_id
            filename = 'input_video.mp4'

            input_file_path = os.path.join(unique_dirname, filename)
            deepgram_audio_file_path = os.path.join(
                unique_dirname, "deepgram_audio.mp3")

            print("Downloading video")
            download_video_start = time.time()
            download_video(url, filename, unique_dirname)
            download_video_end = time.time()
            print(f"""Downloading video took {
                  download_video_end - download_video_start} seconds""")

            if not is_autoposting:
                set_status_for_queued_video(
                    db, queuedVideoId, "Processing 20%")

            deepgram_response = analyze_transcript_deepgram(
                input_file_path, deepgram_audio_file_path)

            deepgram_response_json = json.loads(deepgram_response.to_json())

            words = deepgram_response_json.get("results", {}).get("channels", [{}])[
                0].get("alternatives", [{}])[0].get("words", [])

            if len(words) == 0:
                print(
                    "Deepgram response contains no words. Setting video to isProcessed=True")
                db.video.update(
                    where={
                        "usersVideo": {
                            "userId": user_id,
                            "youtube_id": youtube_video_id,
                            "singleGeneratedVideo": not is_autoposting,
                        }},
                    data={
                        "isProcessed": True
                    }
                )
                if not is_autoposting:
                    set_failed_status_for_queued_video(
                        db, user_id, queuedVideoId)
                return

            if not is_autoposting:
                set_status_for_queued_video(
                    db, queuedVideoId, "Processing 30%")

            identified_moments = identify_moments(deepgram_response_json)

            if not is_autoposting:
                set_status_for_queued_video(
                    db, queuedVideoId, "Processing 40%")

            if len(identified_moments) == 0:
                print(
                    "Claude 3.5 Sonnet found no questions or statements. Setting video to isProcessed=True")
                db.video.update(
                    where={
                        "usersVideo": {
                            "userId": user_id,
                            "youtube_id": youtube_video_id,
                            "singleGeneratedVideo": not is_autoposting,
                        }},
                    data={
                        "isProcessed": True
                    }
                )
                if not is_autoposting:
                    set_failed_status_for_queued_video(
                        db, user_id, queuedVideoId)
                return

            clips = []
            for moment in identified_moments:
                start = moment['start']
                end = moment['end']

                clip = create_and_upload_clip(
                    unique_dirname,
                    input_file_path,
                    start,
                    end,
                    bucket_name,
                    folder_name,
                    db_video_id,
                    user_id
                )

                clips.append(clip)

            db.clip.create_many(data=clips)

            # Save serialized deepgram response to a file
            print("Saving deepgram response to .json file")
            deepgram_response_file_name = "deepgram_response.json"
            deepgram_response_file_path = os.path.join(
                unique_dirname, deepgram_response_file_name)

            with open(deepgram_response_file_path, 'w') as f:
                f.write(deepgram_response.to_json())

            print("Creating metadata.json")
            metadata_file_name = "metadata.json"
            metadata_file_path = os.path.join(
                unique_dirname, metadata_file_name)
            with open(metadata_file_path, 'w') as f:
                json.dump({
                    'long_videos_title': long_videos_title,
                    'long_videos_description': long_videos_description
                }, f)

            s3_client = boto3.client('s3')

            print("Uploading metadata.json")
            s3_client.upload_file(metadata_file_path, bucket_name, f"""{
                folder_name}/metadata.json""")

            print("Uploading transcript.json")
            s3_client.upload_file(deepgram_response_file_path, bucket_name, f"""{
                folder_name}/{deepgram_response_file_name}""")

        if not is_autoposting:
            set_status_for_queued_video(db, queuedVideoId, "Processing 50%")

        if os.environ["ENVIRONMENT"] == "production":
            print("Sending message to the next SQS queue")
            sqs = boto3.client('sqs', region_name=os.environ["AWS_REGION"])

            clip_id = random.choice(clips)['id']
            message_body = json.dumps({
                'clipId': clip_id,
                **({'queuedVideoId': queuedVideoId} if queuedVideoId is not None else {})
            })

            print("Clip id: " + str(clip_id))

            # Send the message
            sqs.send_message(
                QueueUrl=os.environ["SHORTS_CLIPPER_QUEUE_URL"], MessageBody=message_body)

    except RefreshError as e:
        print(f"RefreshError: {e}")

        try:
            oauthToken = db.oauth.find_unique(
                where={
                    "provider_userId": {
                        "userId": user_id,
                        "provider": "youtube"
                    }
                }
            )

            # Don't spam user about invalid token. Only send first time.
            if oauthToken.invalid:
                print("Email about revoked token has already been sent")
                return

            if os.environ["ENVIRONMENT"] == "production":
                send_invalid_yt_oauth_email(
                    user.email, os.environ["RESEND_API_KEY"])

            db.oauth.update(
                where={
                    "provider_userId": {
                        "userId": user_id,
                        "provider": "youtube"
                    }
                },
                data={
                    "invalid": True
                }
            )
        except Exception as e:
            print("Error inside RefreshError:", str(e))
            print(traceback.format_exc())

    except Exception as e:
        print(f"Exception: {e}")
        print(traceback.format_exc())
        if os.environ["ENVIRONMENT"] == "production":
            resend.api_key = os.environ["RESEND_API_KEY"]
            params = {
                "from": "Error detector <errors@useshorts.app>",
                "to": ["errors@useshorts.app"],
                "subject": "Error in video-processor (1st lambda)",
                "html": f"<div>Error is (user_id={user_id}): </strong>{e}</br>{traceback.format_exc()}</div>"
            }

            resend.Emails.send(params)

        try:
            if not is_autoposting:
                set_failed_status_for_queued_video(db, user_id, queuedVideoId)

        except Exception as e:
            print("Error inside Exception:", str(e))
            print(traceback.format_exc())
    finally:
        try:
            if db:
                db.disconnect()
        except Exception as e:
            print("Error while disconnecting database:", str(e))

        if os.environ["ENVIRONMENT"] == "production":
            print("Clean up: deleteting unique_dirname")
            # Always clean temporary files
            if os.path.exists(unique_dirname):
                shutil.rmtree(unique_dirname)


def timeout_handler(context, user_id, queuedVideoId=None):
    print("Function is about to time out. Cleaning up...")

    try:
        send_timeout_email(
            context, os.environ["RESEND_API_KEY"], user_id, queuedVideoId)

        if queuedVideoId != None:
            db = Prisma()
            db.connect()
            set_failed_status_for_queued_video(db, user_id, queuedVideoId)

    except Exception as e:
        print("Error in timeout cleanup: " + str(e))
        print(print(traceback.format_exc()))


def handler(event, context):
    body = json.loads(event['Records'][0]['body'])
    messageId = event['Records'][0]['messageId']
    if "userId" in body:
        timeout_occurred = False

        def timeout_callback():
            nonlocal timeout_occurred
            timeout_occurred = True
            timeout_handler(context, body["userId"],
                            body.get("queuedVideoId"))

        timer = threading.Timer(875, timeout_callback)
        timer.start()

        try:
            if "videoId" in body and "queuedVideoId" in body:
                print("Is user generated video. videoId: " +
                      body["videoId"] + body["queuedVideoId"])
                main(body["userId"], userDefinedVideoId=body["videoId"],
                     queuedVideoId=body["queuedVideoId"])
            else:
                print("Is autoposting: " + body["userId"])
                main(body["userId"])

            return return_success()
        finally:
            timer.cancel()
            if timeout_occurred:
                return return_success()
    else:
        return return_failure(messageId)


if __name__ == '__main__':
    user_id = "clysw0uda0002jovu4idnk78e"

    main(user_id)
