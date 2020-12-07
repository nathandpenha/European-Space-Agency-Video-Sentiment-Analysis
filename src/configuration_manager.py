"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Authors: Vladimir Romashov v.romashov@tue.nl, Hossein Mahdian h.mahdian@tue.nl
@Description: This is a Class to load configurable from yaml configuration file
@Last modified date: 24-11-2020
"""
import yaml
import os


class ConfigurationManager (object):
    """
    ConfigurationManager Class to load configurable from yaml configuration file

    Attributes:
        location: path to the configuration file.
    """
    def __init__(self, location):
        try:
            self.__load_configuration(location)
        except FileNotFoundError:
            # try to find the file in root of project
            current_directory = os.getcwd()
            parent_directory = os.path.dirname(current_directory)
            local_location = str(os.path.join(parent_directory, location))
            self.__load_configuration(local_location)
        print("Current configuration: ")
        print(self.__configuration)

    def __load_configuration(self, location):
        with open(location, 'r') as yamlfile:
            self.__configuration = yaml.load(yamlfile, Loader=yaml.FullLoader)

    @property
    def configuration(self):
        """reads configuration parameters from specified config file (.yml)

        Returns:
            returns a yaml configuration file
        """
        return self.__configuration
