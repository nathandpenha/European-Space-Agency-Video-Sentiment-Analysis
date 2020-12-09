"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Authors: Vladimir Romashov v.romashov@tue.nl, Hossein Mahdian h.mahdian@tue.nl
@Description: This is a Class to load configurable from yaml configuration file
@Last modified date: 08-12-2020
"""
import yaml
import os


class ConfigurationManager(object):
    """
    ConfigurationManager singleton Class to load configurable from yaml configuration file

    Attributes:
        location: path to the configuration file.
    """
    __configuration = None
    __location = 'configuration.yml'

    @staticmethod
    def get_configuration():
        """
        :return: a yaml configuration file
        """
        if ConfigurationManager.__configuration is None:
            ConfigurationManager(ConfigurationManager.__location)
        return ConfigurationManager.__configuration

    def __init__(self, location):
        if ConfigurationManager.__configuration is not None:
            raise Exception("This class is a singleton!")
        try:
            self.__load_configuration(location)
        except FileNotFoundError:
            # try to find the file in root of project
            current_directory = os.getcwd()
            parent_directory = os.path.dirname(current_directory)
            local_location = str(os.path.join(parent_directory, location))
            self.__load_configuration(local_location)

    def __load_configuration(self, location):
        with open(location, 'r') as yamlfile:
            ConfigurationManager.__configuration = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Current configuration: ")
            print(self.__configuration)


if __name__ == '__main__':
    # The code below is for testing / demoing this module stand-alone
    conf1 = ConfigurationManager.get_configuration()
    print(conf1)
    print('****')
    conf2 = ConfigurationManager.get_configuration()
    print(conf2)
    print('****')
    # conf3 = ConfigurationManager("configuration.yml")
    # print(conf3)
