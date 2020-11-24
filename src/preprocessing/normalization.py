"""
Date : 23 November 2020
Authors: Samsom Beyene
"""


import cv2
import numpy as np
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
            face = self.get_frame(face)
            frames.append(face)
        return frames

    def get_frame(self, face):
        revised_face_image = cv2.resize(face, (self.__image_size, self.__image_size))
        revised_face_image = np.array(revised_face_image)
        revised_face_image = revised_face_image / 255
        return revised_face_image

    def save_frames(self, frame_dict, output_path):
        raise NotImplementedError
