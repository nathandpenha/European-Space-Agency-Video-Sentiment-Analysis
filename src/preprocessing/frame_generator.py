"""
Date: 12 November 2020
Authors: Nathan Dpenha, Akram Shokri

This is a preprocessor script that is able to extract image frames from an input video.

The script is able to store the extracted image frames on disk.

The script is able to store the extracted image frames in memory (in a list).

The script is configurable to indicate the number of frames per second that must be extracted
"""

import cv2
import os, sys

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.preprocessing.ipreprocessing import IPreprocessing


class FrameGenerator(IPreprocessing):
    """
    This is a public class used to extract frames from videos.
    """

    __frame_per_second = None
    __CAP_PROP_FRAME_COUNT = 7

    def __init__(self, frame_per_second):
        self.__frame_per_second = frame_per_second

    def get_frames(self, video):
        """
        This function returns a list of frames from a video.
        __CAP_PROP_FRAME_COUNT constant was reassigned locally due to inaccessibility of
        several internal cv2 files from rPI.
        :param video:  video object of type cv2.VideoCapture.
        :return frames: list of frames based on the __frame_per_second parameter.
        """
        video_fps = int(video.get(self.__CAP_PROP_FRAME_COUNT))
        frame_counter = 0
        frames = []
        for i in range(0, video_fps):
            ret, frame = video.read()
            if (frame_counter % self.__frame_per_second == 0):
                frames.append(frame)
            frame_counter += 1
        return frames

    def __count_frames_manual(self, video):
        # initialize the total number of frames read
        total = 0
        # loop over the frames of the video
        while True:
            # grab the current frame
            (grabbed, frame) = video.read()
            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break
            # increment the total number of frames read
            total += 1
        # return the total number of frames in the video file
        return total

    def get_equal_frames(self, video, number_of_frames):
        """
        This function returns a list of frames from a video.
        Number of the returned frames will be equal to number_of_frames.
        __CAP_PROP_FRAME_COUNT constant was reassigned locally due to inaccessibility of
        several internal cv2 files from rPI.
        :param video:  video object of type cv2.VideoCapture.
        :param number_of_frames: an integer identifying number frames to extract
        :return frames: list of frames based on the number_of_frames parameter.
        """
        n_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if int(
                video.get(self.__CAP_PROP_FRAME_COUNT)) > 0 else self.__count_frames_manual(video)
        frames = [x * n_frame / number_of_frames for x in range(number_of_frames)]
        frame_array = []
        for i in range(number_of_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = video.read()
            frame_array.append(frame)
        return frame_array

    def save_frames(self, frame_dict, output_path, number_of_frames=None):
        """
        This function accepts a dictionary of videos with their names and
        extracts and saves frames on the disk.
        :param frame_dict: A dictionary containing the names of videos
        and video objects of type cv2.VideoCapture.
        :param output_path: A path to save the generated frames
        for each video.
        :param number_of_frames: an integer identifying number frames to extract.
        None by default.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename, video in frame_dict.items():
            if number_of_frames is not None:
                frames = self.get_equal_frames(video, number_of_frames)
            else:
                frames = self.get_frames(video)
            for i in range(0, len(frames)):
                if frames[i] is not None:
                    cv2.imwrite(output_path + "/" + filename + "-" + str(i) + ".jpg",
                                frames[i])


def main():
    fg = FrameGenerator(6)
    cap = cv2.VideoCapture("1.mp4")
    frames = fg.get_equal_frames(cap, 10)
    print(frames.__len__())

    cap = cv2.VideoCapture("2.mp4")
    frames = fg.get_equal_frames(cap, 10)
    print(frames.__len__())
    fg.save_frames({"video1": cap}, "./output/", 10)


if __name__ == "__main__":
    main()
