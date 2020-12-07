"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Hossein Mahdian h.mahdian@tue.nl, Vladimir Romashov v.romashov@tue.nl
@Description:
This is an output module script that is able to print emotion destribution from
raspberry camera stream or input video.
The script is able to print output in a file or in command line.
@Last modified date: 30-11-2020
"""

import logging as log
import os, sys
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.configuration_manager import ConfigurationManager


class Logger:
    """
    This class manages logs in console, file, or both
    """
    
    __app_log = None
    __stream_handler = None
    __file_handler = None

    def __init__(self):
        configuration_manager = ConfigurationManager("configuration.yml")
        config = configuration_manager.configuration
        self.__prediction_conf = config['video']['prediction']
        self.__init_logger()

    def __init_logger(self):

        # Get our logger
        self.__app_log = log.getLogger('root')
        self.__app_log.setLevel(log.INFO)
        log_formatter = log.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                       datefmt='%d/%m/%Y %H:%M:%S')

        # Setup Stream Handler (i.e. console)
        self.__stream_handler = log.StreamHandler()
        self.__stream_handler.setFormatter(log_formatter)
        self.__stream_handler.setLevel(log.INFO)
        # File to log to
        if self.__prediction_conf['output_type'] != 'CMD':
            log_file = self.__prediction_conf['log_directory_path'] + "log.log"
            # Setup File handler
            self.__file_handler = log.FileHandler(log_file)
            self.__file_handler.setFormatter(log_formatter)
            self.__file_handler.setLevel(log.INFO)

    def logs(self, result):
        """gets the result of a prediction and log it based on config file

        Args:
            results: list of prediction results
        """

        if self.__prediction_conf['output_type'] == 'CMD':
            self.__logs_cmd(result)
        elif self.__prediction_conf['output_type'] == 'File':
            self.__logs_file(result)
        else:
            self.__logs_cmd(result)
            self.__logs_file(result)

    def __logs_file(self, result):
        max_result = result.max()
        max_position = result.argmax()
        self.__app_log.addHandler(self.__file_handler)
        self.__app_log.info("Emotion distribution set: {}".format(str(result)))
        self.__app_log.info(
            "Detected emotion category : {} probability: {}".format(str(max_position), str(max_result)))
        self.__app_log.info("")

    def __logs_cmd(self, result):
        max_result = result.max()
        max_position = result.argmax()
        # Add the Handler
        self.__app_log.addHandler(self.__stream_handler)
        self.__app_log.info("Emotion distribution set: {}".format(str(result)))
        self.__app_log.info(
            "Detected emotion category : {} probability: {}".format(str(max_position), str(max_result)))

    def info(self, message):
        """gets a Message and add it as info to log

        Args:
            message: information content
        """
        self.__app_log.info(message)

    def error(self, message):
        """gets a Message and add it as error to log

        Args:
            message: information content
        """
        self.__app_log.error(message)
