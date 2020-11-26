"""
Date : 24 November 2020
Authors: Akram Shokri
"""

import cv2
import dlib
import os
import math
import imutils
import numpy as np
from ipreprocessing import IPreprocessing
from frame_generator import FrameGenerator


class SpatialNormalization(IPreprocessing):
    """
    This class is for spatial normalization of faces in frames
    """

    def __init__(self):
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(
            "/Users/Merka/programming/ESA-project/code/video-sentimental-analysis/models/shape_predictor_68_face_landmarks.dat")
        self.__left_eye = np.array([36, 37, 38, 39, 40, 41])
        self.__right_eye = np.array([42, 43, 44, 45, 46, 47])

    def get_frames(self, frame_list):
        """
        This function accepts a list of frames. It detects faces in these frames and normalize them spatially.
        :param frame_list: list of frames each containing a human face
        :return: list of frames with spatially normalized faces
        """
        spatial_normalized_frames = []
        for frame in frame_list:
            spatial_normalized_frames.append(self.__do_spatial_normalization(frame))
        return spatial_normalized_frames

    def save_frames(self, frame_dict, output_path):
        """
        This method accepts a dictionary and an input path. The dictionary contains filename as key
        and the corresponding frame as value. It spatially normalize the face of the subject in all frames and save the
        normalized frames as a .jpg file in the output_path.
        :param frame_dict: A dictionary containing filename as the key and object of type OpenCV frame as values
        :param output_path: A string containing the path for saving the spatially normalized frames
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        frames = []
        images_file_name = []
        for filename, frame in frame_dict.items():
            images_file_name.append(filename)
            frames.append(frame)

        normalized_frames = self.get_frames(frames)
        for i in range(0, len(normalized_frames)):
            if normalized_frames[i] is not None:
                cv2.imwrite(output_path + "/" + images_file_name[i].replace(".jpg", "") + "_spn.jpg",
                            normalized_frames[i])

    def __do_spatial_normalization(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.__detector(gray, 1)

        if len(detections) > 0:
            detected_face = detections[0]
            face_shape = self.__predictor(gray, detected_face)
            landmarks = self.__shape_to_np(face_shape)

            left_eye_center = np.mean(landmarks[self.__left_eye], axis=0)
            right_eye_center = np.mean(landmarks[self.__right_eye], axis=0)
            image = self.__do_alignment(image, left_eye_center, right_eye_center)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.__detector(gray, 1)
            if len(detections) > 0:
                detected_face = detections[0]
                face_shape = self.__predictor(gray, detected_face)
                landmarks = self.__shape_to_np(face_shape)

                left_eye_center = np.mean(landmarks[self.__left_eye], axis=0)
                right_eye_center = np.mean(landmarks[self.__right_eye], axis=0)

                inner_eyes = (right_eye_center[0] - left_eye_center[0]) / 2
                w = inner_eyes * 2.4
                h = inner_eyes * 4.5
                x = left_eye_center[0] + ((right_eye_center[0] - left_eye_center[0]) / 2) - (w / 2)
                y = right_eye_center[1] - (inner_eyes * 1.3)

                image = image[int(y):int(y + h), int(x):int(x + w)]

        return image

    def __shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def __do_alignment(self, img, left_eye, right_eye):
        # this function aligns given face in img based on left and right eye coordinates
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        # find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # find length of triangle edges
        a = self.__find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
        b = self.__find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
        c = self.__find_euclidean_distance(np.array(right_eye), np.array(left_eye))

        # apply cosine rule
        if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / math.pi  # radian to degree

            # rotate base image
            if direction == -1:
                angle = 90 - angle

            img = imutils.rotate(img, direction * angle)

        return img

    def __find_euclidean_distance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance


def main():
    output_path = "/Users/Merka/programming/ESA-project/code/video-sentimental-analysis/src/preprocessing/output/"
    cap = cv2.VideoCapture(
        "/Users/Merka/programming/ESA-project/code/video-sentimental-analysis/src/preprocessing/2.mp4")
    fg = FrameGenerator(6)
    fg.save_frames({"video": cap}, output_path)

    test_frames = {}
    for filename in os.listdir(output_path):
        if ".jpg" in filename:
            image = cv2.imread(output_path + filename)
            test_frames[filename] = image

    sp = SpatialNormalization()
    sp.save_frames(test_frames, output_path + "normalized/")


if __name__ == "__main__":
    main()
