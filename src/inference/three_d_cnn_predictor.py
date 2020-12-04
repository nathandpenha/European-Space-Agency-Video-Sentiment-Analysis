"""
Date: 27 November 2020
Authors: Vladimir Romashov, Hossein Mahdian

This is a Inference module script that is able to detect emotion by 3D CNN models from camera stream or input video.

It uses preprocessing module to prepare date for detection models. Also it uses Output module to display the results.

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
    Emotion inference: takes a video file or Raspberry's camera stream and predicts emotions
    """
    __prediction_conf = None
    __model_blob = None
    __model_type = None
    __output_type = None

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
        if self.__prediction_conf['model_format'] == 'h5':
            self.__spatial_normalizer = SpatialNormalization()
        if self.__prediction_conf['model_format'] == 'IR':
            self.__spatial_normalizer = SpatialNormalization()


    def load_model(self):
        """
        This function:
        - Loads h5 model for predict video emotion
        - Loads IR model for Raspberry camera emotion
        """
        if self.__prediction_conf['model_format'] == 'h5':
            self.model = models.load_model(self.__prediction_conf['model_directory'])
        elif self.__prediction_conf['model_format'] == 'IR':
            self.model, self.model_blob = self.__get_optimized_model()
        else:
            raise ValueError("model format is not correct")

    def __get_optimized_model(self):
        ir_run_conf = self.__prediction_conf['ir_run']
        model_xml = ir_run_conf['model']
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        self.__logger.info("Creating inference Engine")
        self.ie = IECore()
        if ir_run_conf['cpu_extension'] and 'CPU' in ir_run_conf['device']:
            self.ie.add_extension(ir_run_conf['cpu_extension'], "CPU")
        # Read IR
        self.__logger.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = self.ie.read_network(model=model_xml, weights=model_bin)
        if "CPU" in ir_run_conf['device']:
            supported_layers = self.ie.query_network(net, "CPU")
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
        exec_net = self.ie.load_network(network=net, device_name=ir_run_conf['device'])
        return exec_net, model_blob

    def predict_emotion(self):
        """
        This Function predicts the emotion based on its input type
        """
        if self.__is_video_input():
            self.__predict_emotion_video()
        else:
            self.__predict_emotion_webcam()

    def __is_video_input(self):
        """
        This function checks if Inference module is running for prerecorded video
        or Camera
        @:return True if video file was provided
        """
        if self.__prediction_conf['input_type'] == 'Camera':
            return False
        elif self.__prediction_conf['input_type'] == 'Video':
            return True
        raise Exception("input_type should be set to Camera or Video in Configuration")

    def __predict_emotion_video(self):
        video_captor = cv2.VideoCapture(self.__prediction_conf['input_directory'])
        frame_array = self.__get_data_from_video_file(video_captor)
        self.__predict(frame_array)

    def __get_data_from_video_file(self, video_captor):
        """
        This function gets file and returns frames with detected face
        :param file
        :return: frames with detected face
        """
        frame_array_list = []
        if self.__is_video_input():
            d = 0
            if self.__prediction_conf['model_format'] == 'h5':
                n, h, w, d, c = self.model.input_shape
            if self.__prediction_conf['model_format'] == 'IR':
                n, c, h, w, d = self.model.input_info[self.model_blob].input_data.shape
            frame_array_list = self.__prepare_video_3d(video_captor, d)
            return frame_array_list
        else:
            raise Exception("check input type and model format in configuration")

    def __predict_emotion_webcam(self):
        result = None
        camera_stream = cv2.VideoCapture(0)
        camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.__prediction_conf['camera']['CAP_PROP_FRAME_WIDTH'])
        camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__prediction_conf['camera']['CAP_PROP_FRAME_HEIGHT'])
        if self.__prediction_conf['model_format'] == 'h5':
            n, h, w, d, c = self.model.input_shape
        if self.__prediction_conf['model_format'] == 'IR':
            n, c, h, w, d = self.model.input_info[self.model_blob].input_data.shape
        timer = 4
        camera_stream_frames = []
        while True:
            timer = timer + 1
            ret, frame = camera_stream.read()
            cv2.imshow("Frame", frame)
            if timer % 5 == 0:
                camera_stream_frames.append(frame)
            if len(camera_stream_frames) == 12:
                frame_array = self.__prepare_camera_3d(camera_stream_frames)
                if self.__prediction_conf['model_format'] == 'h5':
                    result = self.model.predict(frame_array)
                    self.__logger.logs(result[0])
                    camera_stream_frames = []
                if self.__prediction_conf['model_format'] == 'IR':
                    result = self.model.infer(inputs={self.model_blob: frame_array})
                    for key, probes in result.items():
                        self.__logger.logs(probes[0])
                    camera_stream_frames = []
            if result is not None:
                self.display_result(result, frame)
            else:
                cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __predict(self, frame_array, frame=None):
        if self.__prediction_conf['model_format'] == 'h5':
            result = self.model.predict(frame_array)
            self.display_result(result, frame)
        elif self.__prediction_conf['model_format'] == 'IR':
            ir_result = self.model.infer(inputs={self.model_blob: frame_array})
            self.display_result(ir_result, frame)
        else:
            raise ValueError("model format is not correct")

    def __prepare_image_array(self, image_array):
        image_array = image_array.reshape(1, image_array.shape[0],
                                          image_array.shape[1],
                                          image_array.shape[2], 1)
        if self.__prediction_conf['model_format'] == 'h5':
            image_array = image_array.transpose((0, 2, 3, 1, 4))
        if self.__prediction_conf['model_format'] == 'IR':
            image_array = image_array.transpose((0, 4, 2, 3, 1))
        return image_array

    def __prepare_video_3d(self, cap, d):
        framearray = self.__frame_generator.get_equal_frames(cap, d)
        if self.__prediction_conf['preprocessing']['spatial_normalization']:
            framearray = self.__spatial_normalizer.get_frames(framearray)
        elif self.__prediction_conf['preprocessing']['face_detector']:
            framearray = self.__face_detector.get_frames(framearray)
            if self.__prediction_conf['preprocessing']['face_alignment']:
                framearray = self.__face_alignment.get_frames(framearray)
        framearray = self.__normalizer.get_frames(framearray)
        framearray = np.array(framearray)
        framearray = self.__prepare_image_array(framearray)
        return framearray

    def __prepare_camera_3d(self, framearray):
        if self.__prediction_conf['preprocessing']['spatial_normalization']:
            framearray = self.__spatial_normalizer.get_frames(framearray)
        elif self.__prediction_conf['preprocessing']['face_detector']:
            framearray = self.__face_detector.get_frames(framearray)
            if self.__prediction_conf['preprocessing']['face_alignment']:
                framearray = self.__face_alignment.get_frames(framearray)
        framearray = self.__normalizer.get_frames(framearray)
        framearray = np.array(framearray)
        framearray = self.__prepare_image_array(framearray)
        return framearray

    def display_result(self, result, image):
        """
        This function shows histogram for the rPI camera input and/or provide a command line/text file representation
        of the output data
        :param result: result from prediction
        :param image: captured image
        """
        if not self.__is_video_input():
            if self.__prediction_conf['model_format'] == 'h5':
                self.__gui_output.draw_histogram(result[0], image)
            if self.__prediction_conf['model_format'] == 'IR':
                for key, probes in result.items():
                    ir_result = probes[0]
                    self.__gui_output.draw_histogram(ir_result, image)
        elif self.__is_video_input():
            if self.__prediction_conf['model_format'] == 'h5':
                self.__logger.logs(result[0])
            if self.__prediction_conf['model_format'] == 'IR':
                for key, probes in result.items():
                    ir_result = probes[0]
                    self.__logger.logs(ir_result)


if __name__ == '__main__':
    predict = ThreeDPrediction()
    predict.predict_emotion()
