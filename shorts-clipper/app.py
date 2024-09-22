import glob
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, UTC

import boto3
import cv2
import ffmpegcv
import numpy
import pysubs2
import python_speech_features
import resend
import torch
from dotenv import load_dotenv
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import ResumableUploadError
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

from ASD import ASD
import emails.invalid_yt_oauth
import emails.scheduled
from emails.timeout import send_timeout_email
from model.faceDetector.s3fd import S3FD
from prisma import Prisma
from yunet import YuNet
from deepgram_captions import srt, DeepgramConverter
import random
import threading
import emails.generated
import multiprocessing
from functools import partial

load_dotenv()


def scene_detect(videoPath, pyworkPath):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([videoPath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),
                      videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write('%s - scenes detected %d\n' %
                         (videoPath, len(sceneList)))
    return sceneList


def inference_video_yunet(pyframesPath, videoFilePath, pyworkPath):
    model = YuNet(modelPath="face_detection_yunet_2023mar.onnx",
                  inputSize=[320, 320],
                  confThreshold=0.8,
                  nmsThreshold=0.3,
                  topK=5000,
                  backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                  targetId=cv2.dnn.DNN_TARGET_CPU)

    flist = glob.glob(os.path.join(pyframesPath, '*.jpg'))
    flist.sort()
    dets = []

    # Create a path to save images with detected faces
    output_image_path = os.path.join(pyworkPath, 'detected_faces')
    os.makedirs(output_image_path, exist_ok=True)
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        h, w, _ = image.shape

        # Inference
        model.setInputSize([w, h])
        results = model.infer(image)

        dets.append([])
        for idx, det in enumerate(results):
            bbox = det[0:4].astype(numpy.int32)
            conf = det[-1]
            dets[-1].append({'frame': fidx, 'bbox': [bbox[0],
                            bbox[1], bbox[2], bbox[3]], 'conf': conf})

        # Save the image with detected faces drawn
        output_filename = os.path.join(
            output_image_path, f"frame_{fidx:05d}.jpg")
        cv2.imwrite(output_filename, image)

        sys.stderr.write('%s-%05d; %d dets\r' %
                         (videoFilePath, fidx, len(dets[-1])))

    savePath = os.path.join(pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)

    return dets


def inference_video_s3fd(pyframesPath, videoFilePath, pyworkPath):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cpu')
    flist = glob.glob(os.path.join(pyframesPath, '*.jpg'))
    flist.sort()
    dets = []

    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(
            imageNumpy, conf_th=0.8, scales=[0.1])
        dets.append([])
        for bbox in bboxes:
            # dets has the frames info, bbox info, conf info
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]
                                                     ).tolist(), 'conf': bbox[-1]})

        sys.stderr.write('%s-%05d; %d dets\r' %
                         (videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(sceneFaces, numFailedDet, minTrack, minFaceSize):
    # CPU: Face tracking
    iouThres = 0.5     # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= numFailedDet:
                    iou = bb_intersection_over_union(
                        face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1]+1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2]-bboxesI[:, 0]), numpy.mean(bboxesI[:, 3]-bboxesI[:, 1])) > minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def crop_video(track, cropFile, pyframesPath, cropScale, audioFilePath, nDataLoaderThread, framerate):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(
        pyframesPath, '*.jpg'))  # Read the frames
    flist.sort()
    vOut = ffmpegcv.VideoWriter(
        # Write video
        file=cropFile + 't.avi', codec=None, fps=framerate, resize=(224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:  # Read the tracks
        # dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) # this centers the face
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0])))
        dets['y'].append((det[1]+det[3])/2)  # crop center x
        dets['x'].append((det[0]+det[2])/2)  # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = cropScale
        bs = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                          'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi  # BBox center Y
        mx = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),
                     int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / framerate
    audioEnd = (track['frame'][-1]+1) / framerate
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 48000 -threads %d -ss %.3f -to %.3f %s -loglevel error -hide_banner" %
               (audioFilePath, nDataLoaderThread, audioStart, audioEnd, audioTmp))
    output = subprocess.call(
        command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -r %d -c:a copy %s.avi -loglevel error -hide_banner" %
               # Combine audio and video file
               (cropFile, audioTmp, nDataLoaderThread, framerate, cropFile))
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}


def evaluate_network(files, pretrainModel, pycropPath, framerate):
    # GPU: active speaker detection by pretrained model
    s = ASD()
    s.loadParameters(pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" %
                     pretrainModel)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    # Use this line can get more reliable result
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    for file in files:
        fileName = os.path.splitext(file.split(
            '/')[-1])[0]  # Load audio and video
        audio_file_path = os.path.join(pycropPath, fileName + '.wav')
        audio_file_path_16 = os.path.join(pycropPath, fileName + '_16k.wav')

        cmd = f"ffmpeg -y -i {audio_file_path} -ac 1 -ar 16000 {
            audio_file_path_16} -loglevel error -hide_banner"
        subprocess.call(cmd, shell=True, stdout=None)

        _, audio = wavfile.read(audio_file_path_16)

        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025 * 25 / framerate, winstep=0.010 * 25 / framerate)
        video = cv2.VideoCapture(os.path.join(
            pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112-(112/2)):int(112+(112/2)),
                            int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / (framerate*4), videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * (framerate*4))), :]
        videoFeature = videoFeature[:int(round(length * framerate)), :, :]
        allScore = []  # Evaluation use model
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(
                        audioFeature[i * duration * (framerate*4):(i+1) * duration * (framerate*4), :]).unsqueeze(0).to("cpu")
                    inputV = torch.FloatTensor(
                        videoFeature[i * duration * framerate: (i+1) * duration * framerate, :, :]).unsqueeze(0).to("cpu")
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round(
            (numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


def smooth_face_selection(faces, window_size=5):
    num_frames = len(faces)
    smoothed_faces = []

    for fidx in range(num_frames):
        window_start = max(0, fidx - window_size // 2)
        window_end = min(num_frames, fidx + window_size // 2 + 1)
        window = faces[window_start:window_end]

        # Count how often each track is the highest scoring within the window
        track_counts = {}
        for frame in window:
            if frame:
                max_score_face = max(frame, key=lambda x: x['score'])
                track_counts[max_score_face['track']] = track_counts.get(
                    max_score_face['track'], 0) + 1

        # Select the most consistently highest-scoring track
        if track_counts:
            most_consistent_track = max(track_counts, key=track_counts.get)

            # Use the face data from the most consistent track for the current frame
            current_frame_faces = faces[fidx]
            selected_face = next(
                (face for face in current_frame_faces if face['track'] == most_consistent_track), None)

            if selected_face:
                smoothed_faces.append([selected_face])
            else:
                smoothed_faces.append([])
        else:
            smoothed_faces.append([])

    return smoothed_faces


def render(tracks, scores, pyframesPath, pyaviPath, nDataLoaderThread, output_file, audioFilePath, framerate):
    flist = glob.glob(os.path.join(pyframesPath, '*.jpg'))
    flist.sort()

    faces = [[] for _ in range(len(flist))]
    target_height = 1920
    target_width = 1080

    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            slice = score[max(fidx - 30, 0): min(fidx + 30, len(score))]
            s = numpy.mean(slice) if slice.size else 0
            face_size = track['proc_track']['s'][fidx]
            faces[frame].append({'track': tidx, 'score': s, 's': face_size,
                                 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})

    vOut = None  # Video writer initialization

    # faces = smooth_face_selection(faces)

    for fidx, fname in enumerate(flist):
        original_image = cv2.imread(fname)
        current_faces = faces[fidx]
        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None

        if vOut is None:
            vOut = ffmpegcv.VideoWriter(file=os.path.join(pyaviPath, 'video_only.mp4'),
                                        codec=None, fps=framerate, resize=(target_width, target_height))

        # Determine dynamic mode
        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            scale = target_width / original_image.shape[1]
            resized_height = int(original_image.shape[0] * scale)
            resized_image = cv2.resize(
                original_image, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            # Additional background processing
            scale_for_bg = max(
                target_width / original_image.shape[1], target_height / original_image.shape[0])
            bg_width = int(original_image.shape[1] * scale_for_bg)
            bg_height = int(original_image.shape[0] * scale_for_bg)

            blurred_background = cv2.resize(
                original_image, (bg_width, bg_height))

            blurred_background = cv2.stackBlur(
                blurred_background, ksize=(121, 121))

            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_height - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y +
                                                    target_height, crop_x:crop_x + target_width]

            center_y = (target_height - resized_image.shape[0]) // 2
            blurred_background[center_y:center_y +
                               resized_image.shape[0], :] = resized_image

            vOut.write(blurred_background)

        elif mode == "crop":
            scale = target_height / original_image.shape[0]
            resized_image = cv2.resize(
                original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]
            center_x = int(
                max_score_face['x'] * scale) if max_score_face else frame_width // 2
            top_x = max(min(center_x - target_width // 2,
                        frame_width - target_width), 0)
            image_cropped = resized_image[0:target_height,
                                          top_x:top_x + target_width]

            vOut.write(image_cropped)

    vOut.release()

    command = f"ffmpeg -y -i {os.path.join(pyaviPath, 'video_only.mp4')} -i {audioFilePath} -threads {
        nDataLoaderThread} -c:v copy -c:a aac -ac 2 {output_file} -loglevel error -hide_banner"
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("Error during movie composition:", stderr.decode())

    print("Video render complete:", output_file)


def generate_title_and_description(srt_file_path, long_videos_title, long_videos_description):
    max_retries = 3
    for i in range(max_retries):
        try:
            bedrock = boto3.client(service_name="bedrock-runtime",
                                   region_name="us-east-1")

            if srt_file_path is None:
                transcript_text = ""
            else:
                subs = pysubs2.load(srt_file_path)
                transcript = [sub.text for sub in subs]
                transcript_text = ' '.join(transcript)

            # Get title
            body = json.dumps({
                "max_tokens": 150,
                "messages": [
                    {"role": "user", "content": f"""I have clipped a long youtube video into a short clip. Read the transcript of the short clip and write a really simple title for it without too many fancy words. Make sure it's max 75 characters long. Only output the title and without quotation marks around it, and and dont include any meta comments about the title made by you such as character count. Here is an example output: Ditch free plans for your next startup. Here is the transcript of the short clip:\n{
                        transcript_text} \n\nHere is the title of the long the clip came from video: {long_videos_title} \n\nHere is the description of the long video the clip came from: {long_videos_description}.\n\nUse the headline and description of the original video as a foundation, and only use the transcript of the clip to add more context if the transcript is long enough to make sense. Output in the language that the provided title and description is in."""},
                    {"role": "assistant", "content": "The title is:"}],
                "anthropic_version": "bedrock-2023-05-31"
            })

            title_response = bedrock.invoke_model(
                body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0")

            title_response_body = json.loads(title_response.get("body").read())

            # Get description
            body = json.dumps({
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": f"""I have clipped a long youtube video into a short clip. Read the transcript of the short clip and write a reallysimple description without too many fancy words for the video, with 4-6 relevant hashtags in the end of the description. Make sure its max 500 characters long. Only output the title and without quotation marks around it, and dont include any meta comments about the title made by you such as character count. This is an example: In the digital marketing sphere, the revolution brought about by autoblogging AI software has been nothing short of transformative, especially in the context of SEO writing and achieving top rankings on Google. Here is the transcript of the short clip:\n{
                        transcript_text} \n\nHere is the title of the long the clip came from video: {long_videos_title} \n\nHere is the description of the long video the clip came from: {long_videos_description}.\n\nUse the headline and description of the original video as a foundation, and only use the transcript of the clip to add more context if the transcript is long enough to make sense. Output in the language that the provided title and description is in."""},
                    {"role": "assistant", "content": "The description is:"}],
                "anthropic_version": "bedrock-2023-05-31"
            })

            description_response = bedrock.invoke_model(
                body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0")

            description_response_body = json.loads(
                description_response.get("body").read())

            return title_response_body.get("content")[0]["text"].strip().strip('"').replace("\n", ""), description_response_body.get("content")[0]["text"].strip().strip('"')
        except Exception as e:
            if i < max_retries - 1:  # i is zero indexed
                time.sleep(5)  # wait a bit before trying again
                continue
            else:
                raise


def create_ass_file(srt_file_path, ass_file_path):
    # Load the SRT file
    subs = pysubs2.load(srt_file_path, encoding="utf-8")

    # Adjust the wrap style and resolution
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    # Define the new style
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton" if os.environ.get(
        "ENVIRONMENT") == "production" else "Anton"
    new_style.fontsize = 170
    new_style.primarycolor = "&H00FFFFFF"  # Default white color
    new_style.outline = 5.0
    new_style.shadow = 5.0
    new_style.shadowcolor = "&H80000000"
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 450
    new_style.spacing = -5.0

    subs.styles["Default"] = new_style

    # Function to generate a darker random pastel color
    def random_darker_pastel_color():
        r = random.randint(100, 220)
        g = random.randint(100, 220)
        b = random.randint(100, 220)
        return f"&H{b:02X}{g:02X}{r:02X}"

    # Apply random pastel color to the longest word selectively
    for event in subs.events:
        words = event.text.split()
        if len(words) > 1:  # Check if there is more than one word
            longest_word = max(words, key=len)
            colored_word = f"""{{\\c{random_darker_pastel_color()}}}{
                longest_word}{{\\c&H00FFFFFF&}}"""
            event.text = event.text.replace(longest_word, colored_word, 1)

    # Save as an ASS file
    subs.save(ass_file_path)


def process_transcript(identified_moment_start, identified_moment_end, unique_dirname, deepgram_response):
    print("Filtering words and creating subtitles file")

    clip_dir = os.path.join(unique_dirname, "clip")
    os.makedirs(clip_dir, exist_ok=True)

    srt_file = os.path.join(clip_dir, f"srt.srt")
    ass_file = os.path.join(clip_dir, f"ass.ass")

    # Removes words from utterances outside the paragraph time, and changes the utterance start and end times
    deepgram_response["results"]["utterances"] = [
        {
            "words": [
                {**word, "start": word["start"] - identified_moment_start,
                    "end": word["end"] - identified_moment_start}
                for word in w
            ],
            "start": w[0]["start"] - identified_moment_start,
            "end": w[-1]["end"] - identified_moment_start
        }
        for utterance in deepgram_response["results"]["utterances"]
        if (w := [word for word in utterance["words"] if identified_moment_start <= word["start"] <= identified_moment_end and identified_moment_start <= word["end"] <= identified_moment_end])
    ]

    # Removes all words not inside the paragraph segment
    print("Filter words to clip")
    deepgram_response["results"]["channels"][0]["alternatives"][0]["words"] = [
        word for word in deepgram_response["results"]["channels"][0]["alternatives"][0]["words"]
        if identified_moment_start <= word["start"] <= identified_moment_end
    ]

    # Shift words to be relative to the paragraph start instead of the full video start
    # Shift from e.g {word: "hej", start: 50, end: 51} to {word: "hej", start: 0, end: 1}, because paragraph starts at 50
    print("Adjusting start and end of words")
    for word in deepgram_response["results"]["channels"][0]["alternatives"][0]["words"]:
        word["start"] -= identified_moment_start
        word["end"] -= identified_moment_start

    no_words_in_transcript = len(
        deepgram_response["results"]["channels"][0]["alternatives"][0]["words"]) == 0

    if no_words_in_transcript == False:
        print("Converting to srt")

        deepgram_converter = DeepgramConverter(deepgram_response)
        captions = srt(deepgram_converter, line_length=3)

        # Write the captions to an SRT file
        with open(srt_file, 'w') as f:
            f.write(captions)

        print("Creating ass file")
        create_ass_file(srt_file, ass_file)
    else:
        print("No words found in transcript")

    return clip_dir, no_words_in_transcript


def schedule_video_on_youtube(db: Prisma, file_path, title, description, user_id):
    print("Getting OAuth token")
    oauthToken = db.oauth.find_unique_or_raise(
        where={
            "provider_userId": {
                "userId": user_id,
                "provider": "youtube"
            }
        }
    )

    if oauthToken.invalid == True:
        print("Invalid refresh token for user with id: " + user_id)
        return None

    if oauthToken.selectedChannelId == None:
        print("No channel selected for user with id: " +
              user_id + ". Lambda shouldn't be called.")
        return None

    refresh_token = oauthToken.refreshToken
    post_as_private = True if oauthToken.postAsPrivate else False
    notify_subscribers = oauthToken.notifySubscribers

    client_id = os.getenv('YOUTUBE_CLIENT_ID')
    client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')

    credentials = Credentials.from_authorized_user_info({
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret
    })

    youtube = build('youtube', 'v3', credentials=credentials)

    # Define the media file upload object
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    if len(title) > 100:
        title = title[:100]

    if len(description) > 5000:
        description = description[:5000]

    # Define the video resource
    video_resource = {
        'snippet': {
            'title': title,
            'description': description,
        },
        'status': {
            'privacyStatus': 'private',
            'selfDeclaredMadeForKids': False,
            'notifySubscribers': notify_subscribers,
        },
    }

    if not post_as_private:
        video_resource['status']['publishAt'] = (
            datetime.now(UTC) + timedelta(hours=2)).isoformat()

    print("Scheduling youtube video with title: " + title)

    # Call the videos.insert method to upload and schedule the video
    request = youtube.videos().insert(
        part=','.join(video_resource.keys()),
        body=video_resource,
        media_body=media,
    )
    response = request.execute()

    # Construct the YouTube video URL using the video ID from the response
    video_url = f"https://www.youtube.com/watch?v={response['id']}"

    return video_url


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


def main(clip_id, context=None, queuedVideoId=None):
    is_autoposting = queuedVideoId is None

    try:
        if os.environ["ENVIRONMENT"] == "local":
            temp_dir = "./tmp"
        else:
            temp_dir = "/tmp"

        unique_dirname = os.path.join(
            temp_dir, f"{uuid.uuid4()}".replace('-', ''))
        print("unique_dirname is " + unique_dirname)

        print("Creating tmp and unique dirs")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(unique_dirname, exist_ok=True)

        print("Connecting to db")
        db = Prisma()
        db.connect()
        print("Connected to db")

        print("DB: Looking for clip with id " + clip_id)
        clip = db.clip.find_unique_or_raise(
            where={
                "id": clip_id
            },
            include={"user": True}
        )

        user_id = clip.userId

        if os.environ["ENVIRONMENT"] == "production" and is_autoposting and clip.user.isRunning == False:
            print("UseShorts not running for user with id: " + user_id)
            return

        if not is_autoposting:
            set_status_for_queued_video(db, queuedVideoId, "Processing 60%")

        s3_client = boto3.client('s3')
        bucket_name = os.environ["SHORTS_PROCESSED_VIDEOS"]
        folder_name = str(clip.videoId)

        deepgram_response_file_name = "deepgram_response.json"
        deepgram_response_file_path = os.path.join(
            unique_dirname, deepgram_response_file_name)

        filename = f"{clip.id}.mp4"
        input_file_path = os.path.join(unique_dirname, filename)

        metadata_file_path = os.path.join(unique_dirname, 'metadata.json')

        print("Downloading metadata.json")
        s3_client.download_file(
            bucket_name, f"""{folder_name}/metadata.json""", metadata_file_path)

        print("Downloading transcript.json")
        s3_client.download_file(bucket_name, f"""{
                                folder_name}/{deepgram_response_file_name}""", deepgram_response_file_path)

        print("Downloading clip")
        download_clip_start = time.time()
        s3_client.download_file(
            bucket_name, f"""{folder_name}/{filename}""", input_file_path)
        download_clip_end = time.time()
        print(f"""Downloading clip took {
              download_clip_end - download_clip_start} seconds""")

        with open(metadata_file_path) as f:
            metadata = json.load(f)

        with open(deepgram_response_file_path) as f:
            deepgram_response = json.load(f)

        if not is_autoposting:
            set_status_for_queued_video(db, queuedVideoId, "Processing 70%")

        clip_dir, no_words_in_transcript = process_transcript(
            clip.startTimeInOriginalVideoMs / 1000, clip.endTimeInOriginalVideoMs / 1000, unique_dirname, deepgram_response)

        print("Cropping and adding subtitles")
        nosubs_output_file = os.path.join(
            clip_dir, f'clip_nosubs.mp4')
        output_file = os.path.join(clip_dir,
                                   f'clip.mp4')
        ass_file = os.path.join(clip_dir, f"ass.ass")

        pretrainModel = "weight/pretrain_AVA_CVPR.model"

        nDataLoaderThread = 10
        minTrack = 60
        numFailedDet = 10
        minFaceSize = 1
        cropScale = 0.40

        start = 0

        # Initialization
        pyaviPath = os.path.join(clip_dir, 'pyavi')
        pyframesPath = os.path.join(clip_dir, 'pyframes')
        pyworkPath = os.path.join(clip_dir, 'pywork')
        pycropPath = os.path.join(clip_dir, 'pycrop')
        videoFilePath = os.path.join(pyaviPath, 'video.avi')
        audioFilePath = os.path.join(pyaviPath, 'audio.wav')

        # The path for the input video, input audio, output video
        os.makedirs(pyaviPath, exist_ok=True)
        # Save all the video frames
        os.makedirs(pyframesPath, exist_ok=True)
        # Save the results in this process by the pckl method
        os.makedirs(pyworkPath, exist_ok=True)
        # Save the detected face clips (audio+video) in this process
        os.makedirs(pycropPath, exist_ok=True)

        # Get the framerate of the video
        probe_command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", input_file_path]
        framerate = subprocess.check_output(probe_command).decode().strip()
        print("Video framerate: " + framerate)

        # Evaluate the framerate ratio
        numerator, denominator = map(int, framerate.split('/'))
        framerate = numerator / denominator

        # Cap the framerate to 30 if it's larger than 30
        framerate = min(int(framerate), 60)
        print("Capped framerate: " + str(framerate))

        # Extract video
        command = ["ffmpeg", "-y", "-i", input_file_path, "-qscale:v", "2", "-threads",
                   str(nDataLoaderThread), "-async", "1", "-r", str(framerate), videoFilePath, "-loglevel", "error", "-hide_banner"]

        print("Extracting video")
        extract_video_start = time.time()
        subprocess.call(command, shell=False, stdout=None)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Extract the video and save in %s \r\n" % (videoFilePath))
        extract_video_end = time.time()
        print(f"""Extracting video took {
            extract_video_end - extract_video_start} seconds""")

        print("Extracting audio")
        extract_audio_start = time.time()
        # Extract audio
        command = ["ffmpeg", "-y", "-i", input_file_path, "-qscale:a", "0", "-ac", "1",
                   "-vn", "-threads", str(nDataLoaderThread), "-ar", "48000", audioFilePath, "-loglevel", "error", "-hide_banner"]
        subprocess.call(command, shell=False, stdout=None)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Extract the audio and save in %s \r\n" % (audioFilePath))
        extract_audio_end = time.time()
        print(f"""Extracting audio took {
            extract_audio_end - extract_audio_start} seconds""")

        print("Extracting frames")
        extract_frames_start = time.time()
        # Extract the video frames
        command = ["ffmpeg", "-y", "-i", videoFilePath, "-qscale:v", "2", "-threads",
                   str(nDataLoaderThread), "-start_number", "0", "-f", "image2", os.path.join(pyframesPath, '%06d.jpg'), "-loglevel", "error", "-hide_banner"]
        subprocess.call(command, shell=False, stdout=None)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Extract the frames and save in %s \r\n" % (pyframesPath))
        extract_frames_end = time.time()
        print(f"""Extracting frames took {
            extract_frames_end - extract_frames_start} seconds""")

        print("Detecting scenes")
        scene_detect_start = time.time()
        # Scene detection for the video frames
        scene = scene_detect(videoFilePath, pyworkPath)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Scene detection and save in %s \r\n" % (pyworkPath))
        scene_detect_end = time.time()
        print(f"""Scene detection took {
            scene_detect_end - scene_detect_start} seconds""")

        print("Detecting faces")
        start = time.time()
        faces = inference_video_s3fd(
            pyframesPath, videoFilePath, pyworkPath)
        end = time.time()
        print(f"Detecting faces took {end - start} seconds")
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Face detection and save in %s \r\n" % (pyworkPath))

        print("Tracking faces")
        track_shot_start = time.time()
        # Face tracking
        allTracks, vidTracks = [], []
        for shot in scene:
            # Discard the shot frames less than minTrack frames
            if shot[1].frame_num - shot[0].frame_num >= minTrack:
                # Scene's start and end frames overlap. Below code fixes it.
                # 1: (00:00:00.000 [frame=0, fps=25.000], 00:00:13.640 [frame=341, fps=25.000])
                # 2: (00:00:13.640 [frame=341, fps=25.000], 00:00:18.080 [frame=452, fps=25.000])

                # If not first: +1 start frame
                if shot[0].frame_num != 0:
                    start_frame = shot[0].frame_num + 1
                else:
                    start_frame = shot[0].frame_num

                # +1 end frame
                end_frame = shot[1].frame_num + 1
                faces_in_interval = faces[start_frame:end_frame]

                # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
                allTracks.extend(track_shot(
                    faces_in_interval, numFailedDet, minTrack, minFaceSize))
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Face track and detected %d tracks \r\n" % len(allTracks))
        track_shot_end = time.time()
        print(f"Tracking took {track_shot_end - track_shot_start} seconds")

        print("Cropping to faces")
        crop_video_start = time.time()
        # Face clips cropping
        for ii, track in enumerate(allTracks):
            vidTracks.append(crop_video(
                track, os.path.join(pycropPath, '%05d' % ii), pyframesPath, cropScale, audioFilePath, nDataLoaderThread, framerate))
        savePath = os.path.join(pyworkPath, 'tracks.pckl')
        with open(savePath, 'wb') as fil:
            pickle.dump(vidTracks, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Face Crop and saved in %s tracks \r\n" % pycropPath)
        fil = open(savePath, 'rb')
        vidTracks = pickle.load(fil)
        crop_video_end = time.time()
        print(f"Cropping took {crop_video_end - crop_video_start} seconds")

        print("Active speaker detection")
        active_speaker_detection_start = time.time()
        # Active Speaker Detection
        files = glob.glob("%s/*.avi" % pycropPath)
        files.sort()
        scores = evaluate_network(files, pretrainModel, pycropPath, framerate)
        savePath = os.path.join(pyworkPath, 'scores.pckl')
        with open(savePath, 'wb') as fil:
            pickle.dump(scores, fil)
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                         " Scores extracted and saved in %s \r\n" % pyworkPath)
        active_speaker_detection_end = time.time()
        print(f"""Active speaker detection took {
            active_speaker_detection_end - active_speaker_detection_start} seconds""")

        if not is_autoposting:
            set_status_for_queued_video(db, queuedVideoId, "Processing 80%")

        print("Render clip")
        render_start = time.time()
        render(vidTracks, scores, pyframesPath,
               pyaviPath, nDataLoaderThread, nosubs_output_file, audioFilePath, framerate)
        render_end = time.time()
        print(f"Render took {render_end - render_start} seconds")

        if not is_autoposting:
            set_status_for_queued_video(db, queuedVideoId, "Processing 90%")

        if no_words_in_transcript == False:
            print("Adding subtitles")
            # Add subtitles to clip
            command = (
                f"ffmpeg -i {nosubs_output_file} -vf "
                f"""ass={
                    ass_file}:fontsdir=/usr/share/fonts/truetype/custom -loglevel error -hide_banner {output_file}"""
            )
            subprocess.call(command, shell=True)
            print("Clips with subtitles:", output_file)
        else:
            print("No subtitles to add, renaming nosubs_output_file to output_file")
            os.rename(nosubs_output_file, output_file)

        print("Uploading rendered clip")
        s3_client.upload_file(output_file, os.environ["USESHORTS_CLIPS"],
                              f"{user_id}/{clip.id}.mp4",
                              ExtraArgs={'ContentDisposition': 'attachment'})
        print("Uploaded rendered clip")

        print("Updating clip in db")
        clip = db.clip.update(
            where={
                "id": clip.id,
            },
            data={
                "bucketUrl": f"https://{os.environ["USESHORTS_CLIPS"]}.s3.amazonaws.com/{user_id}/{clip.id}.mp4",
            }
        )

        if (db.clip.count(where={"videoId": clip.videoId, "bucketUrl": None}) == 0):
            print(
                "Cleaning video resources from s3 since all clips from video are rendered")
            s3_resource = boto3.resource('s3')
            bucket = s3_resource.Bucket(bucket_name)
            bucket.objects.filter(Prefix=f"{folder_name}/").delete()

            db.video.update(
                where={
                    "id": clip.videoId
                },
                data={
                    "isProcessed": True
                }
            )

        if os.environ["ENVIRONMENT"] == "production" and is_autoposting:
            long_videos_title = metadata['long_videos_title']
            long_videos_description = metadata['long_videos_description']

            if no_words_in_transcript:
                title, description = generate_title_and_description(
                    None, long_videos_title, long_videos_description)
            else:
                title, description = generate_title_and_description(
                    os.path.join(clip_dir, "srt.srt"), long_videos_title, long_videos_description)

            print("Generated title " + title)
            print("Generated description " + description)

            print("Scheduling to YouTube")
            scheduled_video_link = schedule_video_on_youtube(
                db, output_file, title, description, user_id)

            print("Updating lastScheduledAt")
            user = db.user.update(
                where={
                    "id": user_id
                },
                data={
                    "lastScheduledAt": datetime.now(),
                },
            )

            print("Decrementing credits after clip was autoposted")
            db.user.update(
                where={
                    "id": user_id
                },
                data={
                    "credits": {
                        "decrement": 1
                    }
                }
            )

        if not is_autoposting:
            set_status_for_queued_video(db, queuedVideoId, "Processing 100%")

        user = db.user.find_unique_or_raise(
            where={
                "id": user_id
            },
        )

        if os.environ["ENVIRONMENT"] == "production" and is_autoposting:
            emails.scheduled.send_clip_scheduled_email(
                user.email, os.environ["RESEND_API_KEY"], scheduled_video_link)

        elif os.environ["ENVIRONMENT"] == "production" and not is_autoposting:
            emails.generated.send_clip_generated_email(
                user.email, os.environ["RESEND_API_KEY"], clip.bucketUrl)

            db.queuedvideo.update(
                where={
                    "id": queuedVideoId
                },
                data={
                    "status": "Ready"
                }
            )

    except (RefreshError, ResumableUploadError) as e:
        print(f"RefreshError, ResumableUploadError: {e}")

        try:
            oauthToken = db.oauth.find_unique(
                where={
                    "provider_userId": {
                        "userId": user_id,
                        "provider": "youtube"
                    }
                }
            )

            if 'clip' in locals() and clip is not None:
                emails.generated.send_clip_generated_email(
                    user.email, os.environ["RESEND_API_KEY"], clip.bucketUrl)

            # Don't spam user about invalid token. Only send first time.
            if oauthToken.invalid:
                return

            if os.environ["ENVIRONMENT"] == "production":
                emails.invalid_yt_oauth.send_invalid_yt_oauth_email(
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
            print("Error inside RefreshError, ResumableUploadError:", str(e))
            print(traceback.format_exc())

    except Exception as e:
        try:
            print(f"Exception: {e}")
            print(traceback.format_exc())
            if os.environ["ENVIRONMENT"] == "production":
                log_url = None
                if context is not None:
                    region = context.invoked_function_arn.split(":")[3]
                    log_group_name = context.log_group_name
                    log_stream_name = context.log_stream_name
                    log_url = f"""https://console.aws.amazon.com/cloudwatch/home?region={
                        region}#logEventViewer:group={log_group_name};stream={log_stream_name}"""

                resend.api_key = os.environ["RESEND_API_KEY"]
                params = {
                    "from": "Error detector <errors@useshorts.app>",
                    "to": ["errors@useshorts.app"],
                    "subject": "Error in shorts-clipper (2nd lambda)",
                    "html": f"<div>log url is {str(log_url)}</br>user_id is {user_id})</br>clip_id is {clip_id}</br>Error is {e}</br>{traceback.format_exc()}</div>"
                }

                resend.Emails.send(params)
            if not is_autoposting:
                set_failed_status_for_queued_video(db, user_id, queuedVideoId)
        except Exception as e:
            print("Error while connecting database in Exception:", str(e))
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


def timeout_handler(context, clip_id, queuedVideoId=None):
    print("Function is about to time out. Cleaning up...")

    try:
        send_timeout_email(
            context, os.environ["RESEND_API_KEY"], clip_id)

        if queuedVideoId != None:
            db = Prisma()
            db.connect()
            queuedvideo = db.queuedvideo.find_unique_or_raise(
                where={
                    "id": queuedVideoId
                },
            )
            set_failed_status_for_queued_video(
                db, queuedvideo.userId, queuedVideoId)

    except Exception as e:
        print("Error in timeout cleanup: " + str(e))
        print(print(traceback.format_exc()))


def handler(event, context):
    body = json.loads(event['Records'][0]['body'])
    messageId = event['Records'][0]['messageId']

    if "clipId" in body:
        timeout_occurred = False

        def timeout_callback():
            nonlocal timeout_occurred
            timeout_occurred = True
            timeout_handler(context, body["clipId"],
                            body.get("queuedVideoId"))

        timer = threading.Timer(875, timeout_callback)
        timer.start()

        try:
            if "queuedVideoId" in body:
                print("User generated video. clipId: " +
                      body["clipId"] + ", queuedVideoId: " + body["queuedVideoId"])
                main(body["clipId"], context,
                     queuedVideoId=body["queuedVideoId"])
            else:
                print("Autoposted video: " + body["clipId"])
                main(body["clipId"], context)

            return return_success()
        finally:
            timer.cancel()
            if timeout_occurred:
                return return_success()
    else:
        return return_failure(messageId)


if __name__ == '__main__':
    clip_id = "clzh5kkvt0000chmpw5k8bllg"
    main(clip_id)
