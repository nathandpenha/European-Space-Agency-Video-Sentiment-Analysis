"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Samsom Beyene s.t.beyene@tue.nl
@Contributor: Nathan Dpenha n.z.dpenha@tue.nl
@Description: This script is an interface for the preprocessing submodule
@Last modified date: 17-11-2020
"""

from abc import ABC, abstractmethod


class IPreprocessing(ABC):
    """
    this is an interface class for the preprocessing submodule
    """

    @abstractmethod
    def get_frames(self, frame_list):
        """abstract function

        Args:
            frame_list: list of images stored as type ndarray
        """
        pass

    @abstractmethod
    def save_frames(self, frame_dict, output_path):
        """abstract function

        Args:
            frame_dict: dictionary of frames with their filenames(dictionary)
            output_path: a path to save the detected human faces(string)
        """
        pass
