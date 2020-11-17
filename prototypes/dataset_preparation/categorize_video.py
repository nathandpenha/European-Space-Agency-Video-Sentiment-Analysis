import os
import shutil
import subprocess
import random

raw_data = 'D:\\Projects\\esA\\us10\\Dataset\\prototype'  # replace with path/to/raw/video/dataset
dataset_path = 'D:\\Projects\\esA\\us10\\Dataset\\dataset_v1'  # replace with path/to/categorized/frames/dataset
videos_path = 'D:\\Projects\\esA\\us10\\Dataset\\ravdess_videos'  # replace with path/to/categorized/video/dataset
# temp_path = 'D:\\Projects\\esA\\us10\\Dataset\\temp_videos'

train_size = 0.7
test_size = 0.15
val_size = 0.15


def list_of_emotion(raw_vid_dir, emo):
    """

    :param raw_vid_dir: path to raw video dataset
    :param emo: correlated code for the specific emotion in video's name
    :return: path and names of all video files for a specific emotion
    """
    path = []
    names = []
    for root, dirs, files in os.walk(raw_vid_dir):
        for name in files:
            if name[6:8] == emo:
                path.append(os.path.join(root, name))
                names.append(name)
    return path, names


def extract_frames_from_videos(videos, emo):
    """
    extract frames for all videos of train, test, or val set of specific emotion
    :param videos: a dictionary that it's key specifies train, test, val and value is the list of videos
    :param emo: emotion name
    """
    for video in videos[1]:
        video_name = video[-24: -4]  # taking the name of video file '_.mp4' = -4
        emo_path = os.path.join(dataset_path, videos[0], emo)
        if not os.path.exists(emo_path):
            print(emo_path + " path doesn't exist. trying to make. creating ...")
            os.makedirs(emo_path)
        extract_frames(video, 6, emo_path, emo, video_name)


def extract_frames(input, fps, img_path, emo, video_name):
    """
            extract frames for each video
    :param input: video file
    :param fps: number of frames per second
    :param img_path: path to categorized frames dataset
    :param emo: emotion name
    :param video_name: name of video file
    """
    query = "ffmpeg -f mp4 -i " + input + \
            " -filter:v fps=fps=" + str(fps) + \
            " -f image2 " + os.path.join(img_path, emo + "_frame_%06d_" + video_name + "_.jpg")
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE, ).stdout.read()
    s = str(response).encode('utf-8')


def copy_to_new_dest(dest, emo, file_list):
    """
        copies list of videos related to specific emotion to train set, test set, or val test
    :param dest: path to categorized video dataset
    :param emo: emotion name
    :param file_list: a dictionary that it's key specifies train, test, val and value is the list of videos
    """
    emo_path = os.path.join(dest, file_list[0], emo)
    if not os.path.exists(emo_path):
        print(emo_path + " path doesn't exist. trying to make")
        os.makedirs(emo_path)
    for item in file_list[1]:
        shutil.copy(item, emo_path)


def split_dataset(file_list):
    """

    :param file_list: all the video files of a specific emotion
    :return: all the video files of a specific emotion categorized to train / test / validation sets
    """
    if test_size + val_size + train_size != 1.0:
        raise Exception("Sum of data split is not matching!!")
    data_set = {'train': None, 'validation': None, 'test': None}
    test_samples = random.sample(file_list, int(len(file_list) * test_size))
    remaining_samples = [ele for ele in file_list if ele not in test_samples]
    train_samples = random.sample(remaining_samples, int(len(file_list) * train_size))
    val_samples = [ele for ele in remaining_samples if ele not in train_samples]
    data_set['train'] = train_samples
    data_set['validation'] = val_samples
    data_set['test'] = test_samples
    return data_set


def clean_previous_data():
    """
        removes frames and videos generated from raw video dataset and regenerate them randomly
    """
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        os.makedirs(dataset_path)
    if os.path.exists(videos_path):
        shutil.rmtree(videos_path)
        os.makedirs(videos_path)


def all_videos(root_path, dest_path):
    """

    :param root_path: path to raw video dataset
    :param dest_path: path to a directory to copy all videos of dataset
    """
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file[0:2] == '02':
                path_file = os.path.join(root, file)
                shutil.copy2(path_file, dest_path)  # temp_path


def make_emotion_based_dataset(initial_path, video_dest):
    """

    :param initial_path: path to raw video dataset
    :param video_dest: path to categorized video dataset that
            is going to be divided on emotion and also train/test/validation
    """
    # all_videos(initial_path, temp_path)
    clean_previous_data()
    emotion_dic = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                   '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    for emo_key, emo_value in emotion_dic.items():
        file_list, file_names = list_of_emotion(initial_path, emo_key)
        for data_set in split_dataset(file_list).items():
            copy_to_new_dest(video_dest, emo_value, data_set)
            extract_frames_from_videos(data_set, emo_value)


make_emotion_based_dataset(raw_data, videos_path)
