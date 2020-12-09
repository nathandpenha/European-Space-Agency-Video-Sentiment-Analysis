"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Samsom Beyene s.t.beyene@tue.nl
@Contributor: Hossein Mahdian h.mahdian@tue.nl
@Last modified date: 24-11-2020
"""


import cv2
import numpy as np
import sys
import os
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.preprocessing.ipreprocessing import IPreprocessing


class Normalization(IPreprocessing):
    """
    This class is used to normalizes a list of frames
    """
    def __init__(self, gray_color, image_size):
        self.__gray_color = gray_color
        self.__image_size= image_size

    def get_frames(self, frame_list):
        """returns a list of detected normalized frames

        Args:
            frame_list: list of  normalized images stored as type ndarray(list)

        Returns:
            a list of normalized frames
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
