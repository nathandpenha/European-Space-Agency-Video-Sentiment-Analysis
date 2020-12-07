'''
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Samsom Beyene s.t.beyene@tue.nl
@Contributor: Nathan Dpenha n.z.dpenha@tue.nl
@Description: This script is an interface for the preprocessing submodule
@Last modified date: 17-11-2020
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
    def save_frames(self, frame_dict, output_path):
        '''
        this is an abstract function
        :param frame_dict: dictionary of frames with their filenames
        :type frame_dict: dictionary
        :param output_path: a path to save the detected human faces
        :type output_path: String
        '''
        pass
