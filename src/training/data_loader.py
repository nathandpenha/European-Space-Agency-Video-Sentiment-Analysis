"""
Copyright (c) 2020 TU/e. All rights reserved.
@Author Samsom Beyene s.t.beyene@tue.nl

This script loads the preprocessed training dataset
"""
import os
import numpy as np


class DataLoader:
    """
    This class is used to load a preprocessed train, validation, and test dataset
    """
    def load_train_data(self, data_input_path):
        """
        This function loads and returns the training data
        :param data_input_path: path to  preprocessed training dataset
        :type data_input_path: String
        :return: training data with its labeled classes
        """
        train_data_path = os.path.join(data_input_path, "train_data.npz")
        train_data = np.load(train_data_path)
        x_train, y_train = train_data["data"], train_data["label"]
        return x_train, y_train

    def load_validation_data(self, data_input_path):
        """
        This function loads and returns  validation data
        :param data_input_path: path to the preprocessed validation dataset
        :type data_input_path: String
        :return: validation data with its labeled classes
        """
        validation_data_path = os.path.join(data_input_path, "validation_data.npz")
        validation_data = np.load(validation_data_path)
        x_val, y_val = validation_data["data"], validation_data["label"]
        return x_val, y_val

    def load_test_data(self, data_input_path):
        """
        This function loads and returns  testing data
        :param data_input_path: path to the preprocessed testing dataset
        :type data_input_path: String
        :return: test data with its labeled classes
        """
        test_data_path = os.path.join(data_input_path, "test_data.npz")
        test_data = np.load(test_data_path)
        x_test, y_test = test_data["data"], test_data["label"]
        return x_test, y_test
