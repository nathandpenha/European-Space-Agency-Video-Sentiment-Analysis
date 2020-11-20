import os
import shutil
import subprocess
import random


path = 'D:\\Projects\\esA\\us10\\Dataset\\Ravdess\\Speech'
destination = 'D:\\Projects\\esA\\us10\\Dataset\\ravdess_videos_emotion_based'
frames_dest = 'D:\\Projects\\esA\\us10\\Dataset\\test_data\\frames-emotion_based_test_dataset'


def list_of_emotion(dir, emo):
    path = []
    names = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name[6:8] == emo:
                path.append(os.path.join(root, name))
                names.append(name)
    return path, names


def make_emotion_based_dataset(initial_path, dest):
    emotions = ['01', '02', '03', '04', '05', '06', '07', '08']
    for emo in emotions:
        file_list, file_names = list_of_emotion(initial_path, emo)
        copy_to_new_dest(dest, emo, file_list)
        extract_frames_from_videos(file_list, file_names, emo)


def extract_frames_from_videos(videos, video_names, emo):
    for video in videos:
        video_name = ''
        for vid_name in video_names:
            if video.endswith(vid_name):
                video_name = vid_name
        emo_path = frames_dest + '\\' + emo
        if not os.path.exists(emo_path):
            print(emo_path + " path doesn't exist. trying to make")
            os.makedirs(emo_path)
        extract_frames(video, 6, emo_path, emo, video_name)


def extract_frames(input, fps, img_path, emo, video_name):
    query = "ffmpeg -f mp4 -i " + input + \
            " -filter:v fps=fps=" + str(fps) + \
            " -f image2 " + img_path + "\\" + emo + "_frame_%06d_" + video_name + "_.jpg"
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE, ).stdout.read()
    s = str(response).encode('utf-8')


def copy_to_new_dest(dest, emo, file_list):
    emo_path = dest + '\\' + emo
    if not os.path.exists(emo_path):
        print(emo_path + " path doesn't exist. trying to make")
        os.makedirs(emo_path)
    for item in file_list:
        shutil.copy(item, emo_path)


# def data
# make_emotion_based_dataset(path, destination)
