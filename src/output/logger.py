"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved. 
@Authors: Nathan Dpenha n.z.dpenha@tue.nl, Hossein Mahdian h.mahdian@tue.nl, Vladimir Romashov v.romashov@tue.nl
@Description: This is an output module script that is able to print emotion destribution from
raspberry camera stream or input video.

Last modified date: 07-12-2020
"""
import json
import os, sys
from datetime import datetime
import jsonlines
import operator
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.configuration_manager import ConfigurationManager


class Logger:
    """
    This class manages logs in console, file, or both
    """
    __EMOTIONS = None
    __prediction_conf = None
    __file_name = None
    __log_type = None
    __INFO = "info"
    __ERROR = "error"

    def __init__(self, emotions_list):
        configuration_manager = ConfigurationManager("./../configuration.yml")
        self.__prediction_conf = configuration_manager.configuration['video']['prediction']
        self.__file_name = self.__prediction_conf['log_directory_path'] + "log.json"
        self.__EMOTIONS = emotions_list
        self.__log_type = self.__prediction_conf['output_type']        

    def __combine_results(self, result, emotion_list):
        result = result[0]
        prediction = {}
        for key, value in emotion_list.items():
            prediction[value] = ('{:02.6f}'.format(result[int(key)-1]*100))
        return prediction

    def __format_distribution_result(self, result):
        max_emotion = max(result.items(), key = operator.itemgetter(1))[0]
        max_value = str(max(result.items(), key = operator.itemgetter(1))[1])
        
        reformat_result = {}
        reformat_result["date"] = str(datetime.now())
        reformat_result["log_category"] = self.__INFO
        reformat_result["prediction"] = max_emotion
        reformat_result["value"] = max_value
        reformat_result["all_categories"] = result
        return reformat_result
        
    def __format_error(self, error_msg):
        reformat_error = {}
        reformat_error["date"] = str(datetime.now())
        reformat_error["log_category"] = self.__ERROR
        reformat_error["message"] = error_msg
        return reformat_error
        
    def log_info(self, result):
        """
        This function logs the results of a prediction into the log file specified.
        
        Args:
            result: list of prediction results
        """
        if (self.__log_type is None):
            self.__log_in_file(result, self.__INFO)
            self.__log_on_cmd(result, self.__INFO)
        elif (self.__log_type.lower() == 'file'):
            self.__log_in_file(result, self.__INFO)
        elif (self.__log_type.lower() == 'cmd'):
            self.__log_on_cmd(result, self.__INFO)

    def __log_on_cmd(self, data_to_log, log_category):
        if (log_category == self.__INFO):
            data_to_log = self.__combine_results(data_to_log, self.__EMOTIONS)
            data_to_log = self.__format_distribution_result(data_to_log)
        elif (log_category == self.__ERROR):
            data_to_log = self.__format_error(data_to_log)
        print(json.dumps(data_to_log, indent = 2))

    def log_error(self, err_msg):
        """
        This function logs the error message into the log file specified.
        Args:
            err_msg: String containing the error message  
        """
        if (self.__log_type is None):
            self.__log_in_file(err_msg, self.__ERROR)
            self.__log_on_cmd(err_msg, self.__ERROR)
        elif (self.__log_type.lower() == 'file'):
            self.__log_in_file(err_msg, self.__ERROR)
        elif (self.__log_type.lower() == 'cmd'):
            self.__log_on_cmd(err_msg, self.__ERROR)

    def __log_in_file(self, data_to_log, log_category):
        mode = 'a'
        if (log_category == self.__INFO):
            data_to_log = self.__combine_results(data_to_log, self.__EMOTIONS)
            data_to_log = self.__format_distribution_result(data_to_log)
        elif (log_category == self.__ERROR):
            data_to_log = self.__format_error(data_to_log)
        with jsonlines.open(self.__file_name, mode) as writer:
            writer.write(data_to_log)

# The code below is here for testing / demoing this module stand-alone.
if __name__ == '__main__':
    prediction =   [[9.9981874e-01 ,2.3059013e-12, 4.5604416e-08, 1.8130640e-04, 1.8854964e-08,  9.7346234e-17, 4.4752871e-13, 1.2129508e-11]]
    emotions_list = {"01": "Neutral",
  "02": "Happy" ,
  "03": "Sad" ,
  "04": "Angry" ,
  "05": "Fearful"}
    video_log = Logger(emotions_list)
    video_log.log_info(prediction)
    video_log.log_error("Error message from Main  ")