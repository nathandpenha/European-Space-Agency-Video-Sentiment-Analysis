"""
Date: 19 November 2020
Authors: Vladimir Romashov, Hossein Mahdian

This is a Inference module script that is able to detect emotion from camera stream or input video.

It uses preprocessing module to prepare date for detection models. Also it uses Output module to display the results.

The script is configurable with yaml configuration file
"""
import os
import sys

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

    def __init__(self):
        self.__read_config()
        self.load_model()

    def __read_config(self):
        """
        This function loads configurations from yaml file for prediction module
        """
        configuration_manager = ConfigurationManager("configuration.yml")
        config = configuration_manager.configuration
        self.__prediction_conf = config['video']['prediction']

    def load_model(self):
        """
        This function:
        - Loads CNN or 3d_CNN model for predict emotion
        """
        if self.__prediction_conf['model_type'] == 'CNN':
            self.cnn = CNNPrediction(self.__prediction_conf)
        elif self.__prediction_conf['model_type'] == '3d_CNN':
            self.three_d_cnn = ThreeDPrediction(self.__prediction_conf)
        else:
            raise ValueError("model type is not correct")

    def predict_emotion(self):
        """
        This Function predicts the emotion based on loaded CNN or 3d_CNN model
        """
        if self.__prediction_conf['model_type'] == 'CNN':
            self.cnn.predict_emotion()
        elif self.__prediction_conf['model_type'] == '3d_CNN':
            self.three_d_cnn.predict_emotion()


if __name__ == '__main__':
    predict = Prediction()
    predict.predict_emotion()
