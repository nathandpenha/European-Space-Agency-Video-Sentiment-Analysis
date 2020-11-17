'''
Date: 13 November 2020
Authors: Samsom Beyene
This script is an interface for the preprocessing submodule
'''

from abc import ABC, abstractmethod


class IPreprocessing(ABC):
    """
    this is an interface class for the preprocessing submodule
    """

    @abstractmethod
    def get_frames(self, frame_list):
        '''
        this is an abstract function
        :param frame_list: list of images stored as type ndarray
        :type frame_list: list
        '''
        pass

    @abstractmethod
    def save_frames(self, frames_dict, output_path):
        '''
        this is an abstract function
        :param frames_dict: dictionary of frames with their filenames
        :type frames_dict: dictionary
        :param output_path: a path to save the detected human faces
        :type output_path: String
        '''
        pass
