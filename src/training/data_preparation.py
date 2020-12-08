"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Authors: Samsom Beyene s.t.beyene@tue.nl, Nathan Dpenha n.z.dpenha@tue.nl
@Description:
This script is used to generate training data.
This script preprocess the raw training dataset  and save them into directory.
This class uses a configuration file './config/data_preparation_config.yaml.'
@Last modified date: 04-12-2020
"""

import os
import sys
import cv2
import numpy as np
import yaml
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

current_directory = os.getcwd()  # solves import errors from other submodules
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.preprocessing.frame_generator import FrameGenerator
from src.preprocessing.face_extractor import FaceDetector
from src.preprocessing.normalization import Normalization
from src.preprocessing.face_alignment import FaceAlignment
from src.preprocessing.spatial_normalizer import SpatialNormalization


class DataPreparation:
    """
    This class is used to generate training data
    """
    def get_training_data(self, parameters):
        """gets labeled training frames from videos in each actors directory inside the input directory.

        Args:
            parameters: a dictionary that contains the data preparation parameters(dict)

        Returns:
            list of frames, list of labels(ndarray)
        """
        emotions_dict = parameters["emotion"]
        preprocess_dict = parameters["preprocessing"]
        data_input_path = parameters["data_input_path"]
        frames, labels = [], []
        x_train, y_train = [], []
        x_val, y_val = [], []
        x_test, y_test = [], []
        frame_gen = FrameGenerator(parameters["frame_per_second"])
        face_detect = FaceDetector()
        normalizer = Normalization(parameters["gray_color"], parameters["image_size"])
        face_align = FaceAlignment()
        spatial_normalize = SpatialNormalization()
        dataset_types = self.__read_input_data(data_input_path)
        for dataset in dataset_types:
            emotion_counter = 1
            skipped_emotions = []
            videos = self.__read_input_data(data_input_path + '/' + dataset + '/')
            print("Current preprocessing : " + data_input_path + "/" + dataset + '/')
            for filename in videos:
                video = cv2.VideoCapture(data_input_path + '/' + dataset + '/' + filename)
                if not emotions_dict[filename[6:8]]:
                    if filename[6:8] not in skipped_emotions:
                        emotion_counter += 1
                        skipped_emotions.append(filename[6:8])
                    continue
                video_frames = frame_gen.get_frames(video, parameters["depth"])
                if video_frames is not None or video_frames is not []:
                    for key, value in preprocess_dict.items():
                        if key == "face_detector" and value:
                            video_frames = face_detect.get_frames(video_frames)
                        elif key == "face_alignment" and value:
                            video_frames = face_align.get_frames(video_frames)
                        elif key == "spatial_normalization" and value:
                            video_frames = spatial_normalize.get_frames(video_frames)
                        else:
                            continue
                    if dataset_type == "percentage":
                        frames.append(normalizer.get_frames(video_frames))
                        labels.append(int(filename[6:8]) - emotion_counter)
                    else:
                        if dataset == "training":
                            x_train.append(normalizer.get_frames(video_frames))
                            y_train.append(int(filename[6:8]) - emotion_counter)
                        elif dataset == "validation":
                            x_val.append(normalizer.get_frames(video_frames))
                            y_val.append(int(filename[6:8]) - emotion_counter)
                        else:
                            x_test.append(normalizer.get_frames(video_frames))
                            y_test.append(int(filename[6:8]) - emotion_counter)
        if dataset_type == "percentage":
            return np.array(frames), labels
        else:
            return np.array(x_train), np.array(x_val), np.array(x_test), y_train, y_val, y_test

    def __read_input_data(self, path):
        video_list = []
        for filename in os.listdir(path):
            video_list.append(filename)
        return video_list

    def split_data(self, dataset_type, parameters):
        """splits data into training, validation, and testing dataset

        Args:
            dataset_type: a string used to identify which dataset version to generate(String)
            parameters: a dictionary that contains the data preparation parameters(dict)

        Raises:
            ValueError: wrong input type of dataset_type
        """
        data_output_path = parameters["data_output_path"]
        num_classes = parameters["num_classes"]
        channel = parameters["channel"]
        if dataset_type == "percentage":
            data, labels = self.get_training_data(parameters)
            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.10,
                                                                random_state=42, stratify=labels)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.111,
                                                              random_state=42, stratify=y_train)
        elif dataset_type == "actor":
            x_train, x_val, x_test, y_train, y_val, y_test = self.get_training_data(parameters)

        else:
            raise ValueError("wrong input type{}".format(dataset_type))
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2],
                                   x_train.shape[3], channel))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2],
                               x_val.shape[3], channel))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2],
                                 x_test.shape[3], channel))
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)

        if not os.path.exists(data_output_path):
            print("directory {} does not exists.".format(data_output_path))
            os.makedirs(data_output_path)
            print("created directory {}".format(data_output_path))

        np.savez_compressed(data_output_path + "/train_data", data=x_train, label=y_train)
        np.savez_compressed(data_output_path + "/validation_data", data=x_val, label=y_val)
        np.savez_compressed(data_output_path + "/test_data", data=x_test, label=y_test)
        print("Preprocessed data saved successfully")


if __name__ == '__main__':
    with open("config/data_preparation_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    dataset_type = config["dataset_type"]
    parameters = config["data_parameters"]
    data_prepare = DataPreparation()
    data_prepare.split_data(dataset_type, parameters)
