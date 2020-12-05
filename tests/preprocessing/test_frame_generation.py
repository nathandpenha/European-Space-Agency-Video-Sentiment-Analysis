"""
Copyright (c) 2020 TU/e -PDEng Software Technology c2019. All rights reserved.
@Author: Vladimir Romashov v.romashov, Georgios Azis g.azis@tue.nl
Description:
This is a test frame generation module script that is able to test get_frame, get_frames,
get_equal_frames, and save_frames methods.

Preprocessing.frame_generator module; os, shutil, yaml, numpy, and opencv2 libraries were
used in this script.
A folder with videos with and without faces was created as the initial test data.

Last modified date: 02-12-2020
"""

import cv2
import os
import shutil
import yaml
import numpy as np
from src.preprocessing.frame_generator import FrameGenerator

project_path = os.path.abspath(os.path.join(__file__, "../../.."))
videos_path = project_path + '/prod_data/tests/test_videos/'


def test_frame_generation_different_images():
    """
    This method checks if we are getting a list of different frames from get_frames.
    :return The test is complete if the first and the last images are not equal.
    """
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    frames = frame_generator.get_frames(video)
    for i in range(len(frames) - 1):
        assert not (np.array_equal(frames[i], frames[i+1]))


def test_frame_generation_empty_input():
    """
    This method checks the behaviour of get_frames method for a nonexistent input
    video file.
    :return The test is complete if the output is empty.
    """
    video = cv2.VideoCapture(videos_path + 'invalid_file.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    frames_from_frame_generator = frame_generator.get_frames(video)
    assert len(frames_from_frame_generator) == 0


def test_get_equal_frames_with_correct_input():
    """
    This method checks if the number of the returned frames in get_frames will be
    equal to the predefined number_of_frames.
    :return The test is complete if the get_equal_frames number of output frames is
    equal to the number_of_frames.
    """
    number_of_frames = 10
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    frames_from_frame_generator = frame_generator.get_frames(video, number_of_frames)
    assert len(frames_from_frame_generator) == number_of_frames


def test_get_equal_frames_with_incorrect_input():
    """
    This method checks the behavior of get_frames with predefined number of frames
    for a nonexistent input video file.
    :return The test is complete if the output of get_equal_frames is empty.
    """
    number_of_frames = 10
    video = cv2.VideoCapture(videos_path + 'invalid_file.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    frames_from_frame_generator = frame_generator.get_frames(video, number_of_frames)
    empty_output = True
    for frame in frames_from_frame_generator:
        if frame is not None:
            empty_output = False
    assert empty_output == True


def test_get_equal_frames_with_empty_frames():
    """
    This method checks the behaviour of get_frames with the number of frames
    equal to zero.
    :return The test is complete if the output of get_equal_frames is empty.
    """
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    frames_from_frame_generator = frame_generator.get_frames(video, 0)
    assert len(frames_from_frame_generator) == 0


def test_get_equal_frames_with_negative_number_of_frames():
    """
    This method checks the behaviour of get_frames with the negative number
    of frames.
    :return The test is complete if the error is raised.
    """
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    raised = True
    try:
        _ = frame_generator.get_frames(video, -1)
        raised = False
    except:
        pass
    finally:
        assert raised == True


def test_save_frames():
    """
    This method checks weather the save_frames method is able to save faces in the existing folder
    with the correct path.
    :return The test is complete if the path of the generated frames is not empty.
    """
    generated_frames_path = project_path + '/prod_data/tests/test_images/generated_frames/'
    __clean_folder(generated_frames_path)
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    video_2 = cv2.VideoCapture(videos_path + 'video_with_face_2.mp4')
    videos_dict = {"video": video, "video_2": video_2}
    frame_generator = __get_frame_generator_with_frame_per_second()
    frame_generator.save_frames(videos_dict, generated_frames_path)
    assert len(os.listdir(generated_frames_path)) != 0


def test_save_frames_with_the_number_of_frames():
    """
    This method tests two things:
    - weather the save_frames method is able to save faces in the existent folder with
    the correct path.
    - whether the number of saved faces matches the number of original frames.
    :return The test is complete if the path of the generated frames contains number of
    frames equals to the number_of_frames.
    """
    number_of_frames = 5
    generated_frames_path = project_path + '/prod_data/tests/test_images/generated_frames/'
    __clean_folder(generated_frames_path)
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    frame_generator.save_frames({"video": video}, generated_frames_path, number_of_frames)
    number_of_created_files = len([name for name in os.listdir(generated_frames_path)
                                   if os.path.isfile(os.path.join(generated_frames_path, name))])
    assert number_of_created_files == number_of_frames


def test_save_frames_with_new_path():
    """
    This method checks weather the save_frames method is able to save faces in the non-existing folder
    with the correct path.
    :return The test is complete if the path of the generated frames is not empty.
    """
    generated_frames_path = project_path + '/prod_data/tests/test_images/generated_frames_new_folder/'
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    frame_generator.save_frames({"video": video}, generated_frames_path)
    assert len(os.listdir(generated_frames_path)) != 0


def test_save_frames_with_wrong_path():
    """
    This method checks the behaviour of save_frames with an empty string as a path direction.
    :return The test is complete if the error is not raised.
    """
    generated_frames_path = ''
    video = cv2.VideoCapture(videos_path + 'video_with_face.mp4')
    frame_generator = __get_frame_generator_with_frame_per_second()
    raised = True
    try:
        _ = frame_generator.save_frames({"video": video}, generated_frames_path)
        raised = False
    finally:
        assert raised == False


def __get_frame_generator_with_frame_per_second():
    configuration_manager = __load_configuration()
    prediction_conf = configuration_manager['video']['prediction']
    return FrameGenerator(prediction_conf['frame_per_second'])


def __load_configuration():
    file_path = project_path + '/src/configuration.yml'
    with open(file_path, 'r') as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)


def __clean_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
