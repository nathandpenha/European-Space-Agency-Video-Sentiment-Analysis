"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@ Authors: Akram Shokri a.shokri@tue.nl
@ Contributors: Yusril Maulidan Raji y.m.raji@tue.nl
Last modified date: 01-12-2020
"""

import os
import math
import cv2
import dlib
import numpy as np
from utility import Utility
from ipreprocessing import IPreprocessing
from frame_generator import FrameGenerator


class SpatialNormalization(IPreprocessing):
    """
    This class is for extracting spatial normalization part of face from image(s)
    """
    def __init__(self, rpi=False):
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(
            "../../models/shape_predictor_68_face_landmarks.dat")
        self.__left_eye = np.array([36, 37, 38, 39, 40, 41])
        self.__right_eye = np.array([42, 43, 44, 45, 46, 47])

        self.__is_rpi = rpi
        if self.__is_rpi:
            # initialize openvino face detection lib
            from openvino.inference_engine import IECore
            ie = IECore()
            net_face = ie.read_network(model="../../models/face-detection-adas-0001.xml",
                                       weights="../../models/face-detection-adas-0001.bin")
            self.__exec_net_face = ie.load_network(network=net_face, device_name="MYRIAD")
            self.__input_blob_face = next(iter(net_face.input_info))
            self.__output_blob_face = next(iter(net_face.outputs))
            n_face, c_face, self.__h_face, self.__w_face = net_face.input_info[self.__input_blob_face].input_data.shape

    def __detect_face_openvino(self, image):
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.__w_face, self.__h_face))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        output = self.__exec_net_face.infer(inputs={self.__input_blob_face: image})
        output = output[self.__output_blob_face]
        box = output[0][0][0][3:] * np.array([width, height, width, height])
        x_min, y_min, x_max, y_max = box.astype(np.int32)

        return x_min, y_min, x_max, y_max

    def get_frames(self, frame_list):
        """
        This function accepts a list of frames. It detects faces in these frames and normalize them spatially.
        :param frame_list: list of frames each containing a human face
        :return: list of frames with spatially normalized faces
        """
        spatial_normalized_frames = []
        for frame in frame_list:
            # resize image to decrease spatial normalization execution time
            img_height = frame.shape[0]
            resize_percent = Utility.calculate_resize_percent(img_height)
            if resize_percent < 1:
                frame = Utility.resize_image(frame, resize_percent)
            # collect spatial normalized frame
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
        face_shape = None
        if self.__is_rpi:
            # if rpi, use openvino model to detect the face
            x_min, y_min, x_max, y_max = self.__detect_face_openvino(image)
            face_shape = self.__predictor(gray, dlib.rectangle(x_min, y_min, x_max, y_max))
        else:
            # if non rpi, use dlib model to detect the face
            detections = self.__detector(gray, 1)
            if len(detections) > 0:
                detected_face = detections[0]
                face_shape = self.__predictor(gray, detected_face)

        if face_shape is not None:
            landmarks = self.__shape_to_np(face_shape)

            left_eye_center = np.mean(landmarks[self.__left_eye], axis=0)
            right_eye_center = np.mean(landmarks[self.__right_eye], axis=0)
            image, left_eye_center, right_eye_center = self.__do_alignment(image, left_eye_center, right_eye_center)

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

        c = self.__find_euclidean_distance(np.array(right_eye), np.array(left_eye))

        new_left_eye = left_eye
        new_right_eye = right_eye

        # find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
            new_right_eye = (left_eye_x + c, left_eye_y)
            center_rotate = left_eye
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
            new_left_eye = (right_eye_x - c, right_eye_y)
            center_rotate = right_eye

        # find length of triangle edges
        a = self.__find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
        b = self.__find_euclidean_distance(np.array(right_eye), np.array(point_3rd))

        # apply cosine rule
        if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / math.pi  # radian to degree

            # rotate base image
            if direction == -1:
                angle = 90 - angle

            img_mat = cv2.getRotationMatrix2D((center_rotate[0], center_rotate[1]), direction * angle, 1.0)
            img = cv2.warpAffine(img, img_mat, (img.shape[1], img.shape[0]))

        return img, new_left_eye, new_right_eye

    def __find_euclidean_distance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance


def main():
    output_path = "./output/"
    cap = cv2.VideoCapture(
        "./2.mp4")
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
