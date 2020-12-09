"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Hossein Mahdian h.mahdian@tue.nl, Vladimir Romashov v.romashov@tue.nl
@Description:
This is a Inference module script that is able to detect emotion from camera stream or input video.
It uses preprocessing module to prepare date for detection models. Also it uses Output module to display the results.
The script is configurable with yaml configuration file
@Last modified date: 30-11-2020
"""
import os
import sys
import yaml

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.inference.configuration_manager import ConfigurationManager
from src.inference.cnn_predictor import CNNPrediction
from src.inference.three_d_cnn_predictor import ThreeDPrediction


class Prediction:
    """
    Emotion inference: takes a video file or camera stream and predicts emotions
    """
    __prediction_conf = None
    __cnn = None
    __three_d_cnn = None
    __enabled_emotions = None

    def __init__(self):
        self.__read_config()
        self.__read_enabled_emotions()
        self.load_model()

    def __read_config(self):
        """
        This function loads configurations from yaml file for prediction module
        """
        config = ConfigurationManager.get_configuration()
        self.__prediction_conf = config['video']['prediction']

    def __read_enabled_emotions(self):
        with open("../training/config/data_preparation_config.yaml") as file:
            config = yaml.load(file, Loader = yaml.FullLoader)
        emotions_dict = config['data_parameters']['emotion']
        enabled_emotions = []
        for key, value in emotions_dict.items():
            enabled_emotions.append(value)
        self.__enabled_emotions = enabled_emotions

    def load_model(self):
        """
        This function:
        - Loads CNN or 3d_CNN model for predict emotion
        """
        if self.__prediction_conf['model_type'].lower() == 'cnn':
            self.__cnn = CNNPrediction(self.__prediction_conf, self.__enabled_emotions)
        elif self.__prediction_conf['model_type'].lower() == '3d_cnn':
            self.__three_d_cnn = ThreeDPrediction(self.__prediction_conf, self.__enabled_emotions)
        else:
            raise ValueError("model type is not correct")

    def predict_emotion(self):
        """
        This Function predicts the emotion based on loaded CNN or 3d_CNN model
        """
        if self.__prediction_conf['model_type'].lower() == 'cnn':
            self.__cnn.predict_emotion()
        elif self.__prediction_conf['model_type'].lower() == '3d_cnn':
            self.__three_d_cnn.predict_emotion()


if __name__ == '__main__':
    predict = Prediction()
    predict.predict_emotion()
