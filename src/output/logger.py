"""
Date: 19 November 2020
Authors: Hossein Mahdian, Vladimir Romashov

This is an output module script that is able to print emotion destribution from
raspberry camera stream or input video.

The script is able to print output in a file or in command line.
"""

import logging as log
import os
import sys
from colorlog import ColoredFormatter

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.inference.configuration_manager import ConfigurationManager


class Logger:
    """
    This class manages logs in console, file, or both
    """

    __app_log = None
    __stream_handler = None
    __file_handler = None
    __emotion_dict = {0: "Neutral",
                      1: "Happy",
                      2: "Sad",
                      3: "Angry",
                      4: "Fearful"}

    def __init__(self):
        config = ConfigurationManager.get_configuration()
        self.__prediction_conf = config['video']['prediction']
        self.__init_logger()

    def __init_logger(self):

        # Get our logger
        self.__app_log = log.getLogger('root')
        self.__app_log.setLevel(log.INFO)
        log_format = "  %(message)s%(reset)s"
        c_format = '%(log_color)s' + log_format
        colors = {'DEBUG': 'green',
                  'INFO': 'bold_cyan',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_blue'}
        log_formatter = ColoredFormatter(c_format, log_colors=colors)

        # Setup Stream Handler (i.e. console)
        self.__stream_handler = log.StreamHandler()
        self.__stream_handler.setFormatter(log_formatter)
        self.__stream_handler.setLevel(log.INFO)
        # File to log to
        if self.__prediction_conf['output_type'].lower() == 'file':
            log_file = self.__prediction_conf['log_directory_path'] + "log.log"
            # Setup File handler
            self.__file_handler = log.FileHandler(log_file)
            self.__file_handler.setFormatter(log_formatter)
            self.__file_handler.setLevel(log.INFO)

    def logs(self, result):
        """
        gets the result of a prediction and log it based on config file
        :param output_type:
        :param result:
        """
        if self.__prediction_conf['output_type'].lower() == 'cmd':
            self.__logs_cmd(result)
        elif self.__prediction_conf['output_type'].lower() == 'file':
            self.__logs_file(result)
        else:
            self.__logs_cmd(result)
            self.__logs_file(result)

    def __logs_file(self, result):
        max_position = result.argmax()
        self.__app_log.addHandler(self.__file_handler)
        self.__app_log.info("\n\t ==== Video model prediction result ====\n")
        self.__app_log.info("\t Emotion \t Probability")
        for i in range(5):
            self.__app_log.info("\t {} : \t {}".format(str(self.__emotion_dict[i]).ljust(7), str(result[i])))
        self.__app_log.critical(
            "\n\t Detected emotion : {}, probability: {}".format(str(self.__emotion_dict[max_position]),
                                                                 str((result[max_position]))))
        self.__app_log.debug("Inserted...")

    def __logs_cmd(self, result):
        max_position = result.argmax()
        self.__app_log.addHandler(self.__stream_handler)
        self.__app_log.info("\n\t ==== Video model prediction result ====\n")
        self.__app_log.info("\t Emotion \t Probability")
        for i in range(5):
            self.__app_log.info("\t {} : \t {}".format(str(self.__emotion_dict[i]).ljust(7), str(result[i])))
        self.__app_log.critical(
            "\n\t Detected emotion : {}, probability: {}".format(str(self.__emotion_dict[max_position]),
                                                                 str((result[max_position]))))

    def info(self, message):
        """
        gets a Message and add it as info to log
        :param message:
        """
        self.__app_log.info(message)

    def error(self, message):
        """
        gets a Message and add it as error to log
        :param message:
        """
        self.__app_log.error(message)
