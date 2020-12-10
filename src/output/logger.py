"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Authors: Nathan Dpenha n.z.dpenha@tue.nl, Hossein Mahdian h.mahdian@tue.nl, Vladimir Romashov v.romashov@tue.nl
@Description: This is an output module script that is able to print emotion destribution from
raspberry camera stream or input video.
Last modified date: 10-12-2020
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
    __emotions = None

    def __init__(self):
        config = ConfigurationManager.get_configuration()
        self.__prediction_conf = config['video']['prediction']
        self.__init_logger()

    def __init_logger(self):
        self.__emotions = self.__prediction_conf['emotion_map']
        # Get our logger
        self.__app_log = log.getLogger('root')
        self.__app_log.setLevel(log.INFO)
        log_format = "  %(message)s%(reset)s"
        c_format = '%(log_color)s' + log_format
        json_cmd_formatter = "{\"time\": \"%(asctime)s\", \"level\": \"%(log_color)s%(levelname)s\", \"distribution\": %(log_color)s%(message)s\" }"
        json_file_formatter = log.Formatter(
            "{\"time\": \"%(asctime)s\", \"level\": \"%(levelname)s\", \"distribution\": %(message)s\" }")
        colors = {'DEBUG': 'green',
                  'INFO': 'bold_cyan',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_blue'}
        cmd_log_formatter = ColoredFormatter(json_cmd_formatter, log_colors=colors)

        # Setup Stream Handler (i.e. console)
        self.__stream_handler = log.StreamHandler()
        self.__stream_handler.setFormatter(cmd_log_formatter)
        self.__stream_handler.setLevel(log.INFO)
        # File to log to
        if self.__prediction_conf['output_type'] is not None and \
                self.__prediction_conf['output_type'].lower() == 'file':
            log_file = self.__prediction_conf['log_directory_path'] + "log.json"
            if not os.path.exists(log_file):
                open(log_file, "w+")
            self.__file_handler = log.FileHandler(log_file)
            self.__file_handler.setFormatter(json_file_formatter)
            self.__file_handler.setLevel(log.INFO)

    def logs(self, result):
        """
        gets the result of a prediction and log it based on config file
        :param output_type:
        :param result:
        """
        if self.__prediction_conf['output_type'] is not None and self.__prediction_conf['output_type'].lower() == 'cmd':
            self.__logs_cmd(result)
        elif self.__prediction_conf['output_type'] is not None and self.__prediction_conf[
            'output_type'].lower() == 'file':
            self.__logs_file(result)
        else:
            self.__logs_cmd(result)
            self.__logs_file(result)

    def __logs_file(self, result):
        self.__app_log.addHandler(self.__file_handler)
        dist_of_emotions = {}
        for index, emotion in enumerate(self.__emotions):
            dist_of_emotions[self.__emotions[emotion]] = result[index]
        max_key = max(dist_of_emotions, key=dist_of_emotions.get)
        json_emotion_dict = (str(dist_of_emotions)).replace('\'', '\"')
        self.__app_log.info(
            "{} , {}".format(json_emotion_dict, str('\"detected emotion\": \"' + max_key)))

    def __logs_cmd(self, result):
        self.__app_log.addHandler(self.__stream_handler)
        dist_of_emotions = {}
        for index, emotion in enumerate(self.__emotions):
            dist_of_emotions[self.__emotions[emotion]] = result[index]
        max_key = max(dist_of_emotions, key=dist_of_emotions.get)
        json_emotion_dict = (str(dist_of_emotions)).replace('\'', '\"')
        self.__app_log.info(
            "{} , {}".format(json_emotion_dict, str('\"detected emotion\": \"' + max_key)))

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
