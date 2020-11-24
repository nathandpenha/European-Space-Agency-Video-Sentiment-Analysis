import yaml
import os
import cv2
from os import sys
current_directory = os.getcwd()  # solves import errors from other submodules
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.preprocessing.frame_generator import FrameGenerator
from src.preprocessing.face_extractor import FaceDetector
from src.preprocessing.normalization import Normalization
from src.preprocessing.face_alignment import FaceAlignment


class DataPreparation:
    __data_input_path = None
    __data_output_path = None
    __dataset_type = None
    __frames_per_second = None
    __depth = None
    __face_detector = None
    __face_alignment = None
    __image_gray = None

    def __init__(self, config_file):
        self.__data_input_path = config["data_input_path"]
        self.__data_output_path = config["data_output_path"]
        self.__dataset_type = config["dataset_type"]
        self.__frame_per_second = config["frame_per_second"]
        self.__depth = config["depth"]
        self.__image_gray = config["image_gray"]
        self.__image_width = config["image_width"]
        self.__image_height = config["image_height"]
        self.__face_detector = config["preprocessing"][0]
        self.__face_alignment = config["preprocessing"][1]

    def prepare_training_data(self):
        labels = []
        frames = []
        frame_gen = FrameGenerator(self.__frames_per_second)
        face_detect = FaceDetector()
        normalizer = Normalization( self.__image_gray, self.__image_height)
        face_align = FaceAlignment()
        actors = self.__get_raw_data(self.__data_input_path)
        print(actors)
        for actor in actors:
            videos = self.__get_raw_data(self.__data_input_path + actor + '/')
            print("Current preprocessing : " + self.__data_input_path + actor + '/')
            for filename in videos:
                print(filename)
                video = cv2.VideoCapture(self.__data_input_path + actor + '/' + filename)
                video_frames = frame_gen.get_equal_frames(video, self.__depth)
                if video_frames is not None or video_frames is not []:
                    face_frames = face_detect.get_frames(video_frames)
                    align_frames = face_align.get_frames(face_frames)
                    frames.append(normalizer.get_frames(align_frames))
                    labels.append(int(filename[6:8]) - 1)
        return np.array(frames), labels

    def __get_raw_data(self, path):
        video_list = []
        for filename in os.listdir(path):
            video_list.append(filename)
        return video_list


if __name__ == '__main__':
    config = {}
    with open("data_preparation_config.yaml") as file:
        config = yaml.load(file)
        # print(config)
    datapreparation = DataPreparation(config)
    datapreparation.prepare_training_data()
