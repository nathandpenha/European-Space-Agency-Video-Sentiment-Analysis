"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Samsom Beyene s.t.beyene@tue.nl
@Description:
This is a training module that  is used to train a given ANN type architecture.
This script loads   training, validation, and testing dataset.
This script builds, trains, and evaluates  a  cnn or 3D cnn model type.
This script saves the trained model into a given directory.
This scripts save  trained model information into a given directory.
This class uses a configuration file './config/training_config.yaml'
@Last modified date: 27-11-2020
"""

import os
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import yaml
import numpy as np
from data_loader import DataLoader
from model_builder import MobileNet
from model_builder import ThreeDCNNV1


class Training:
    """
    This class  is used to built, train and evaluate a (3D)CNN type of model
    """

    def __init__(self, learning_rate):
        self.__opt = Adam(learning_rate=learning_rate)

    def load_data(self, input_path):
        """
        This function returns the training, validation and testing dataset with their labels
        :param input_path: a path to lead the preprocessed training dataset
        :type input_path: String
        :return: a pair of training, validation, and testing dataset
        :rtype: list
        """
        data_loader = DataLoader()
        x_train, y_train = data_loader.load_train_data(input_path)
        x_val, y_val = data_loader.load_validation_data(input_path)
        x_test, y_test = data_loader.load_test_data(input_path)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def __convert_3d_to_2d_array(self, images, labels, gray_color):
        """
        This function returns a 2d array list of gray images with their class labels
        :param images: array of images
        :type images: numpy array
        :param labels: array of labels
        :type labels: numpy array
        :return: array of gray images with their labels
        :rtype: numpy array
        """
        x_new = []
        y_new = []
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                if gray_color:
                    x_new.append(images[i, j, :, :, 0])
                else:
                    x_new.append(images[i, j, :, :, :])
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                y_new.append(labels[i, :])
        if gray_color:
            return np.repeat(np.array(x_new)[..., np.newaxis], 3, -1), np.array(y_new)
        else:
            return np.array(x_new), np.array(y_new)

    def __save_training_info(self, model_type, parameters, training_history):
        print("=== Saving training informations ==== ")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(script_dir, "training_log")
        yaml_file_path = os.path.join(log_path, parameters["trained_model_name"] + ".yaml")
        print(yaml_file_path)
        parameters['model_input_shape'] = list(parameters['model_input_shape'])
        yaml_content = {
            "model_type": model_type,
            "parameters": parameters
            }
        with open(yaml_file_path, 'w+') as f:
            yaml.dump(yaml_content, f)
        history_file_path = os.path.join(log_path, parameters["trained_model_name"] + "_history.npy")
        np.save(history_file_path, training_history.history)

    def evaluate_model(self, model, x_test, y_test):
        """
        This function evaluates and displays the performance of the model on test data.
        :param model: a trained model
        :type model: float
        :param x_test: array list of frames
        :type x_test: numpy array
        :param y_test: list of labels for frames
        :type y_test: numpy array
        """
        model_score = model.evaluate(x_test, y_test, verbose=1)
        print("Model accuracy: {}".format(model_score[1]))

    def train_model(self, model_type, parameters):
        """
        This function trains and evaluates a given cnn model based training parameters.
        :param model_type: type of model architecture
        :param parameters: a list of training parameters
        :type parameters: dict
        """
        model_dict = {"MobileNet": MobileNet, "ThreeDCNNV1": ThreeDCNNV1}
        # load data
        x_train, y_train, x_val, y_val, x_test, y_test = self.load_data(parameters["data_input_path"])
        print("======= Loaded dimensions ========")
        print("Training dimension", x_train.shape, y_train.shape)
        print("Validation dimensions", x_val.shape,y_val.shape)
        print("Testing dimensions", x_test.shape, y_test.shape)
        if model_type == "cnn":
            x_train, y_train = self.__convert_3d_to_2d_array(x_train, y_train, parameters["gray_color"])
            x_val, y_val = self.__convert_3d_to_2d_array(x_val, y_val, parameters["gray_color"])
            x_test, y_test = self.__convert_3d_to_2d_array(x_test, y_test, parameters["gray_color"])
        if model_type == "3d_cnn":
            x_train = x_train.transpose((0, 2, 3, 1, 4))
            x_val = x_val.transpose((0, 2, 3, 1, 4))
            x_test = x_test.transpose((0, 2, 3, 1, 4))
        print(" ======= Postprocessing dimensions ========= ")
        print("Training dimension", x_train.shape, y_train.shape)
        print("Validation dimensions", x_val.shape, y_val.shape)
        print("Testing dimensions", x_test.shape, y_test.shape)
        # create model
        if parameters["model_builder"] in model_dict:
            model_builder = model_dict[parameters["model_builder"]]()
        else:
            raise ValueError("model builder {} does not exist. Please select from:{}".
                             format(parameters["model_builder"], model_dict))

        model = model_builder.build_model(parameters["num_classes"], parameters["n_conv_filters"], parameters["kernel_size"], parameters["model_input_shape"], parameters["depth"])
        print(model.summary())
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
        mc = ModelCheckpoint(parameters["model_output_path"] + '/' + parameters["trained_model_name"],
                             monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        model.compile(optimizer=self.__opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print("========Training starts==========")
        model_history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                                  batch_size=parameters["batch_size"], epochs=parameters["epochs"],
                                  verbose=1, shuffle=True, callbacks=[mc, es])

        print("=======evaluating model's test accuracy============")
        self.evaluate_model(model, x_test, y_test)
        self.__save_training_info(model_type, parameters, model_history)
        print("=====Training finished=====")


if __name__ == '__main__':
    with open("config/training_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    model_type = config["model_type"]
    train_parameters = config["train_params"]
    lr = train_parameters["learning_rate"]
    trainer = Training(lr)
    trainer.train_model(model_type, train_parameters)
