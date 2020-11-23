"""
Date : 23 November 2020
Authors: Samsom Beyene
"""


import cv2
from .ipreprocessing import IPreprocessing


class Normalization(IPreprocessing):
    """
    This class is used to normalizes a list of frames
    """
    def __init__(self, gray_color, image_size):
        self.__gray_color = gray_color
        self.__image_size= image_size

    def get_frames(self, frame_list):
        """
        This function returns a list of detected normalized frames

        :param frame_list: list of  normalized images stored as type ndarray
        :type frame_list: list
        :return: a list of normalized frames
        :rtype: list
        """
        frames = []
        for face in frame_list:
            if self.__gray_color:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (self.__image_size, self.__image_size))
            face = face / 255
            frames.append(face)
        return frames

    def save_frames(self, frame_dict, output_path):
        raise NotImplementedError
