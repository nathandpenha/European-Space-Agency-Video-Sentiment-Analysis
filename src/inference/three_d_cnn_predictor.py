"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Hossein Mahdian h.mahdian@tue.nl, Vladimir Romashov v.romashov@tue.nl
@Description:
This is a Inference module script that is able to detect emotion by 3D CNN models from camera stream or input video.
It uses preprocessing module to prepare date for detection models. Also it uses Output module to display the results.
@Last modified date: 07-12-2020
"""
import os
import sys

import cv2
import numpy as np
from tensorflow.keras import models

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.output.logger import Logger
from src.output.gui_output import GUIOutput
from src.preprocessing.face_extractor import FaceDetector
from src.preprocessing.face_alignment import FaceAlignment
from src.preprocessing.frame_generator import FrameGenerator
from src.preprocessing.normalization import Normalization
from src.preprocessing.spatial_normalizer import SpatialNormalization
from openvino.inference_engine import IECore


class ThreeDPrediction:
    """
    3D Emotion inference: takes a video file or camera stream and predicts emotions using 3D CNN
    """
    __prediction_conf = None
    __model_blob = None
    __model = None
    __model_type = None
    __output_type = None
    __ie = None

    def __init__(self, predict_conf):
        self.__prediction_conf = predict_conf
        self.__frame_generator = FrameGenerator(self.__prediction_conf['frame_per_second'])
        self.__face_detector = FaceDetector()
        self.__face_alignment = FaceAlignment()
        self.__normalizer = Normalization(self.__prediction_conf['gray_color'],
                                          self.__prediction_conf['model_input_shape'][1]['height'])
        self.__logger = Logger()
        self.__gui_output = GUIOutput()
        self.load_model()
        if self.__prediction_conf['model_format'].lower() == 'h5':
            self.__spatial_normalizer = SpatialNormalization()
        if self.__prediction_conf['model_format'].lower() == 'ir':
            self.__spatial_normalizer = SpatialNormalization(self.__ie, True)

    def load_model(self):
        """
        This function:
        - Loads h5 model and IR model for predict emotion
        """
        if self.__prediction_conf['model_format'].lower() == 'h5':
            self.__model = models.load_model(self.__prediction_conf['model_directory'])
        elif self.__prediction_conf['model_format'].lower() == 'ir':
            self.__model, self.__model_blob = self.__get_optimized_model()
        else:
            raise ValueError("model format is not correct")

    def __get_optimized_model(self):
        ir_run_conf = self.__prediction_conf['ir_run']
        model_xml = ir_run_conf['model']
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        self.__logger.info("Creating inference Engine")
        self.__ie = IECore()
        if ir_run_conf['cpu_extension'] and 'CPU' in ir_run_conf['device']:
            self.__ie.add_extension(ir_run_conf['cpu_extension'], "CPU")
        # Read IR
        self.__logger.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = self.__ie.read_network(model=model_xml, weights=model_bin)
        if "CPU" in ir_run_conf['device']:
            supported_layers = self.__ie.query_network(net, "CPU")
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                self.__logger.error(
                    "Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(ir_run_conf['device'], ', '.join(not_supported_layers)))
                self.__logger.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        model_blob = next(iter(net.input_info))
        self.__logger.info("Loading model to the plugin")
        exec_net = self.__ie.load_network(network=net, device_name=ir_run_conf['device'])
        return exec_net, model_blob

    def predict_emotion(self):
        """
        This Function predicts the emotion based on its input type
        """
        if self.__is_video_input():
            self.__predict_emotion_video()
        else:
            self.__predict_emotion_camera_stream()

    def __is_video_input(self):
        """
        This function checks if Inference module is running for prerecorded video
        or Camera
        @:return True if video file was provided
        """
        if self.__prediction_conf['input_type'].lower() == 'camera':
            return False
        elif self.__prediction_conf['input_type'].lower() == 'video':
            return True
        raise Exception("input_type should be set to Camera or Video in Configuration")

    def __predict_emotion_video(self):
        video_captor = cv2.VideoCapture(self.__prediction_conf['input_directory'])
        frame_array = self.__get_data_from_video_file(video_captor)
        self.__predict(frame_array)

    def __get_data_from_video_file(self, video_captor):
        """
        This function gets file and returns frames based on preprocessing configs
        :param video_captor: a video file containing a human face with emotion
        :return: preprocessed frames ready for inference
        """
        depth = 0
        if self.__prediction_conf['model_format'].lower() == 'h5':
            n, height, width, depth, color = self.__model.input_shape
        if self.__prediction_conf['model_format'].lower() == 'ir':
            n, color, height, width, depth = self.__model.input_info[self.__model_blob].input_data.shape
        frame_array = self.__frame_generator.get_number_of_frames(video_captor, depth, self.__prediction_conf['is_rpi'])
        frame_array_list = self.__preprocess_frames(frame_array)
        return frame_array_list

    def __predict_emotion_camera_stream(self):
        result = None
        depth = 0
        camera_stream = cv2.VideoCapture(0)
        camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.__prediction_conf['camera']['CAP_PROP_FRAME_WIDTH'])
        camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__prediction_conf['camera']['CAP_PROP_FRAME_HEIGHT'])
        if self.__prediction_conf['model_format'].lower() == 'h5':
            n, height, width, depth, color = self.__model.input_shape
        if self.__prediction_conf['model_format'].lower() == 'ir':
            n, color, height, width, depth = self.__model.input_info[self.__model_blob].input_data.shape
        timer = 4
        camera_stream_frames = []
        while True:
            timer = timer + 1
            ret, frame = camera_stream.read()
            text_frame = frame.copy()
            if ret and frame is not None:
                cv2.imshow("Frame", text_frame)
                if timer % 5 == 0:
                    camera_stream_frames.append(frame)
                if len(camera_stream_frames) == depth:
                    frame_array = self.__preprocess_frames(camera_stream_frames)
                    if self.__is_frame_array_ready(frame_array, depth):
                        if self.__prediction_conf['model_format'].lower() == 'h5':
                            result = self.__model.predict(frame_array)
                            self.__logger.logs(result[0])
                        if self.__prediction_conf['model_format'].lower() == 'ir':
                            result = self.__model.infer(inputs={self.__model_blob: frame_array})
                            for key, probes in result.items():
                                self.__logger.logs(probes[0])
                    camera_stream_frames = []
                if result is not None:
                    self.display_result(result, text_frame)
                else:
                    cv2.imshow("Frame", text_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __predict(self, frame_array, frame=None):
        if self.__prediction_conf['model_format'].lower() == 'h5':
            result = self.__model.predict(frame_array)
            self.display_result(result, frame)
        elif self.__prediction_conf['model_format'].lower() == 'ir':
            ir_result = self.__model.infer(inputs={self.__model_blob: frame_array})
            self.display_result(ir_result, frame)
        else:
            raise ValueError("model format is not correct")

    def __prepare_image_array(self, image_array):
        image_array = image_array.reshape(1, image_array.shape[0],
                                          image_array.shape[1],
                                          image_array.shape[2], 1)
        batch_dim = 0
        color_dim = 1
        height_dim = 2
        width_dim = 3
        depth_dim = 4
        if self.__prediction_conf['model_format'] == 'h5':
            image_array = image_array.transpose((batch_dim, height_dim, width_dim, color_dim, depth_dim))
        if self.__prediction_conf['model_format'] == 'IR':
            image_array = image_array.transpose((batch_dim, depth_dim, height_dim, width_dim, color_dim))
        return image_array

    def __preprocess_frames(self, frame_array):
        if self.__prediction_conf['preprocessing']['spatial_normalization']:
            frame_array = self.__spatial_normalizer.get_frames(frame_array)
        elif self.__prediction_conf['preprocessing']['face_detector']:
            frame_array = self.__face_detector.get_frames(frame_array)
            if self.__prediction_conf['preprocessing']['face_alignment']:
                frame_array = self.__face_alignment.get_frames(frame_array)
        frame_array = self.__normalizer.get_frames(frame_array)
        frame_array = np.array(frame_array)
        frame_array = self.__prepare_image_array(frame_array)
        return frame_array

    def display_result(self, result, image):
        """
        This function shows histogram for the rPI camera input and/or provide a command line/text file representation
        of the output data
        :param result: result from prediction
        :param image: captured image
        """
        if not self.__is_video_input():
            if self.__prediction_conf['model_format'].lower() == 'h5':
                self.__gui_output.draw_histogram(result[0], image)
            if self.__prediction_conf['model_format'].lower() == 'ir':
                for key, probes in result.items():
                    ir_result = probes[0]
                    self.__gui_output.draw_histogram(ir_result, image)
        elif self.__is_video_input():
            if self.__prediction_conf['model_format'].lower() == 'h5':
                self.__logger.logs(result[0])
            if self.__prediction_conf['model_format'].lower() == 'ir':
                for key, probes in result.items():
                    ir_result = probes[0]
                    self.__logger.logs(ir_result)

    def __is_frame_array_ready(self, frame_array, depth):
        ir_depth_dimension = frame_array.shape[-1]
        h5_depth_dimension = frame_array.shape[-2]
        if self.__prediction_conf['model_format'].lower() == 'ir':
            return ir_depth_dimension == depth
        if self.__prediction_conf['model_format'].lower() == 'h5':
            return h5_depth_dimension == depth


if __name__ == '__main__':
    """The code below is for testing /
     demoing this module stand-alone"""
    predict = ThreeDPrediction()
    predict.predict_emotion()
