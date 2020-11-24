"""
Date: 19 November 2020
Authors: Vladimir Romashov, Hossein Mahdian

This is a Inference module script that is able to detect emotion from camera stream or input video.

It uses preprocessing module to prepare date for detection models. Also it uses Output module to display the results.

The script is configurable with yaml configuration file
"""
import cv2
import numpy as np
from tensorflow.keras import models
import os, sys

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.configuration_manager import ConfigurationManager
from src.Output.logger import Logger
from src.Output.gui_output import GUIOutput
from src.preprocessing.face_extractor import FaceDetector
from src.preprocessing.frame_generator import FrameGenerator
from src.preprocessing.normalization import Normalization


class Prediction:
    """
    Emotion inference: takes a video file or Raspberry's camera stream and predicts emotions
    """
    __prediction_conf = None
    __model_blob = None
    __model_type = None
    __output_type = None
    __mode = None  # RPi, Webcam (for demo), Record, Image (optional)
    __logger = None

    def __init__(self):
        self.__read_config()

    def __read_config(self):
        """
        This function loads configurations from yaml file for prediction module
        """
        configuration_manager = ConfigurationManager("configuration.yml")
        config = configuration_manager.configuration
        self.__prediction_conf = config['video']['prediction']
        self.__logger = Logger()
        self.__gui_output = GUIOutput()
        self.load_model()
        self.__frame_generator = FrameGenerator(self.__prediction_conf['frame_per_second'])
        self.__face_detector = FaceDetector()
        self.__normalizer = Normalization(False, self.__prediction_conf['model_input_shape'][1]['height'])

    def display_result(self, result, image):
        """
        This function shows histogram for the rPI camera input and/or provide a command line/text file representation
        of the output data
        :param result: result from prediction
        :param image: captured image
        """

        if self.__is_raspberry():

            for key, probes in result.items():
                ir_result = probes[0]
                self.__gui_output.draw_histogram(ir_result, image)
                self.__logger.logs(ir_result)
        else:
            self.__logger.logs(result)

    def __is_raspberry(self):
        """
        This function checks if inference module is running on Raspberry's camera stream
        or is it getting webcam stream
        @:return True if it is running on Raspberry
        """
        if self.__prediction_conf['mode'] == 'RPi':
            is_raspberry = True
        else:
            is_raspberry = False
        return is_raspberry

    def __is_video_input(self):
        """
        This function checks if inference module is running for prerecorded video
        or not
        @:return True if video file was provided
        """
        if self.__prediction_conf['input_directory'] is None:
            is_video_input = False
        else:
            is_video_input = True
        return is_video_input

    def get_data(self, file):
        """
        This function get camera frame or video and returns normalized frames with detected faces
        :param file: file can be a frame or video
        :return:  frame or frames with detected face
        """
        if self.__is_raspberry():
            image = file
            face_image = self.__face_detector.get_frame(image)
            if face_image is not None:
                normalized_face = self.__normalizer.get_frame(face_image)
                return normalized_face
        # path to the folder with video
        if not self.__is_raspberry():
            video = file
            frames = self.__frame_generator.get_frames(video)
            faces = self.__face_detector.get_frames(frames)
            normalized_faces = self.__normalizer.get_frames(faces)
            return normalized_faces

    def __image_reshape(self, image):
        return image.reshape(1, self.__prediction_conf['model_input_shape'][1]['height'],
                             self.__prediction_conf['model_input_shape'][2]['width'],
                             self.__prediction_conf['model_input_shape'][0]['channels'])

    def load_model(self):
        """
        This function:
        - Loads h5 model for predict video emotion
        - Loads IR model for Raspberry camera emotion
        """
        if self.__prediction_conf['mode'] == 'Record':
            self.__model = models.load_model(self.__prediction_conf['model_directory'])
        elif self.__prediction_conf['mode'] == 'RPi':
            self.__model, self.__model_blob = self.__get_optimized_model()

    def predict_emotion(self):
        """
        This Function predicts the emotion based on its input type
        """
        if self.__is_video_input():
            self.__predict_emotion_video()
        elif self.__is_raspberry():
            self.__predict_emotion_rpi_camera()
        else:
            self.__predict_emotion_webcam()

    def __predict_emotion_video(self):
        video = cv2.VideoCapture(self.__prediction_conf['input_directory'])
        normalized_faces = self.get_data(video)
        for normal_face in normalized_faces:
            normalized_face_image = self.__image_reshape(normal_face)
            result = self.__model.predict(normalized_face_image)
            self.display_result(result, normal_face)

    def __predict_emotion_rpi_camera(self):
        # get openvino loaded model
        optimized_model, input_blob = self.__model, self.__model_blob
        # initialize the camera and grab a reference to the raw camera capture
        camera, rawCapture = self.__get_rPI_cam_input()
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array
            normalized_face = self.get_data(image)
            if normalized_face is not None:
                prepared_face_image = self.__prepare_image(normalized_face, optimized_model, input_blob)
                ir_result = optimized_model.infer(inputs={input_blob: prepared_face_image})
                self.display_result(ir_result, image)
            cv2.imshow("Frame", image)
            rawCapture.truncate(0)
            key = cv2.waitKey(1) & 0xFF
            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    def __get_rPI_cam_input(self):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(640, 480))
        return camera, rawCapture

    def __get_optimized_model(self):
        from openvino.inference_engine import IECore
        ir_run_conf = self.__prediction_conf['ir_run']
        model_xml = ir_run_conf['model']
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        self.__logger.info("Creating inference Engine")
        ie = IECore()
        if ir_run_conf['cpu_extension'] and 'CPU' in ir_run_conf['device']:
            ie.add_extension(ir_run_conf['cpu_extension'], "CPU")
        # Read IR
        self.__logger.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = ie.read_network(model=model_xml, weights=model_bin)
        if "CPU" in ir_run_conf['device']:
            supported_layers = ie.query_network(net, "CPU")
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                self.__logger.error(
                    "Following layers are not supported by the plugin for specified device {}:\n {}".
                    format(ir_run_conf['device'], ', '.join(not_supported_layers)))
                self.__logger.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        input_blob = next(iter(net.input_info))
        self.__logger.info("Loading model to the plugin")
        exec_net = ie.load_network(network=net, device_name=ir_run_conf['device'])
        return exec_net, input_blob

    def __prepare_image(self, image, net, input_blob):
        n, c, h, w = net.input_info[input_blob].input_data.shape
        images = np.ndarray(shape=(n, c, h, w))
        for i in range(n):
            if image.shape[:-1] != (h, w):
                image = cv2.resize(image, (w, h))
            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            images[i] = image
        return images


if __name__ == '__main__':
    predict = Prediction()
    predict.predict_emotion()
