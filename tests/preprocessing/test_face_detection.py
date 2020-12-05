"""
Copyright (c) 2020 TU/e -PDEng Software Technology c2019. All rights reserved.
@Author: Vladimir Romashov v.romashov, Georgios Azis g.azis@tue.nl
Description:
This is a test face extraction module script that is able to test get_frame, get_frames,
and save_frames methods.

Preprocessing.FaceDetector module; os, shutil, and opencv2 libraries were used in this script.
A folder with face images and a folder with images without faces were created as initial
test data.

Last modified date: 02-12-2020
"""

import cv2
import os
import shutil
from src.preprocessing.face_extractor import FaceDetector

__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
__face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
__face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
__face_cascade3 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
__face_cascade4 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

project_path = os.path.abspath(os.path.join(__file__, "../../.."))
videos_path = project_path + '/prod_data/tests/test_videos/'


def test_face_extractor_image_with_face():
    """
    This method checks get_frame for an image with a face.
    :return The test is complete if we can detect a face on the output image of the face extractor.
    """
    assert __is_face_on_the_photo(__get_first_frame_with_face_detection('with_face')) is not None


def test_face_extractor_image_without_face():
    """
    This method checks get_frame for an image without a face.
    :return The test is complete if we cannot detect a face on the output image of the face extractor.
    """
    assert __is_face_on_the_photo(__get_first_frame_with_face_detection('without_face')) is None


def test_face_extractor_number_of_frames_with_faces():
    """
    This method checks if get_frames method detects faces for multiple images with faces.
    :return The test is complete if the number of detected faces is equal to the number of
    input images with faces.
    """
    frames = __get_list_of_frames_from_folder('with_face')
    assert len(frames) == len(__get_faces_from_frames(frames))


def test_face_extractor_number_of_frames_without_faces():
    """
    This method checks if get_frames method returns nothing for multiple images
    without faces.
    :return The test is complete if the number of detected faces are equal to zero.
    """
    frames = __get_list_of_frames_from_folder('without_face')
    assert len(__get_faces_from_frames(frames)) == 0


def test_save_frames():
    """
    This method tests two things:
    - weather the save_frames method is able to save processed images in the existing folder with
    the correct path.
    - whether the number of saved images matches the number of original files.
    :return The test is complete if the number of detected faces is equal to the number of
    input frames.
    """
    generated_faces_path = project_path + '/prod_data/tests/test_images/generated_faces/'
    __clean_folder(generated_faces_path)
    frames = __get_dict_of_frames_from_folder('with_face')
    face_extractor = FaceDetector()
    face_extractor.save_frames(frames, generated_faces_path)
    number_of_created_files = len([name for name in os.listdir(generated_faces_path)
                                   if os.path.isfile(os.path.join(generated_faces_path, name))])
    assert number_of_created_files == len(frames)


def test_save_frames_with_new_path():
    """
    This method tests two things:
    - weather the save_frames method is able to save processed images in the non-existent folder with
    the correct path.
    - whether the number of saved images matches the number of original files.
    :return The test is complete if the number of detected faces is equal to the number of
    input frames.
    """
    generated_faces_path = project_path + '/prod_data/tests/test_images/generated_faces_new_folder/'
    frames = __get_dict_of_frames_from_folder('with_face')
    face_extractor = FaceDetector()
    face_extractor.save_frames(frames, generated_faces_path)
    number_of_created_files = len([name for name in os.listdir(generated_faces_path)
                                   if os.path.isfile(os.path.join(generated_faces_path, name))])
    assert number_of_created_files == len(frames)


def test_save_frames_empty_list():
    """
    This method checks the behaviour of save_frames for an empty dictionary of images.
    :return  The test is complete if the path with the generated faces is empty.
    """
    generated_faces_path = project_path + '/prod_data/tests/test_images/generated_faces/'
    __clean_folder(generated_faces_path)
    frames_empty = {}
    face_extractor = FaceDetector()
    face_extractor.save_frames(frames_empty, generated_faces_path)
    number_of_created_files = len([name for name in os.listdir(generated_faces_path)
                                   if os.path.isfile(os.path.join(generated_faces_path, name))])
    assert number_of_created_files == 0


def test_save_frames_empty_list_with_wrong_path():
    """
    This method checks the behaviour of save_frames with the following inputs:
    - empty dictionary of images
    - empty string as a path direction
    :return  The test is complete if the error is not raised.
    """
    wrong_generated_faces_path = ''
    frames_empty = {}
    face_extractor = FaceDetector()
    raised = True
    try:
        _ = face_extractor.save_frames(frames_empty, wrong_generated_faces_path)
        raised = False
    finally:
        assert raised == False


def __get_first_frame_with_face_detection(folder_name):
    frames = __get_list_of_frames_from_folder(folder_name)
    face_extractor = FaceDetector()
    return face_extractor.get_frame(frames[0])


def __get_faces_from_frames(frames):
    face_extractor = FaceDetector()
    return face_extractor.get_frames(frames)


def __get_list_of_frames_from_folder(folder_name):
    frames = []
    path = project_path + '/prod_data/tests/test_images/' + folder_name
    for filename in os.listdir(path):
        full_file_name = os.path.join(path, filename)
        frames.append(cv2.imread(full_file_name))
    return frames


def __get_dict_of_frames_from_folder(folder_name):
    image_dict = {}
    path = project_path + '/prod_data/tests/test_images/' + folder_name
    for filename in os.listdir(path):
        full_file_name = os.path.join(path, filename)
        image_dict[os.path.basename(filename)] = cv2.imread(full_file_name)
    return image_dict


def __is_face_on_the_photo(frame):
    face_image = None
    face = __face_cascade.detectMultiScale(frame, 1.2, 5)
    if len(face) == 0 or len(face) > 1:
        face = __face_cascade1.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
        if len(face) == 0 or len(face) > 1:
            face = __face_cascade2.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
            if len(face) == 0 or len(face) > 1:
                face = __face_cascade3.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
                if len(face) == 0 or len(face) > 1:
                    face = __face_cascade4.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
    for (x, y, w, h) in face:
        face_image = frame[y:y + h, x:x + w]
    return face_image


def __clean_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
