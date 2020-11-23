"""
Date: 20 November 2020
Authors: Samsom Beyene, Nathan Dpenha, Akram Shokri

This is a training module that  is used to train a given cnn type architecture.

This script is able to read training dataset from directory.

This script calls the preprocessing module for processing the training data.

This script is able to split the training data into training, validation, and testing.

This script builds, trains, and evaluates  a mobilenet cnn model.

This script saves the trained model into into a given directory.
"""

import os
import sys
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import yaml
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

current_directory = os.getcwd()  # solves import errors from other submodules
parent_directory = os.path.dirname(current_directory)
grand_parent_directory = os.path.dirname(parent_directory)
sys.path.insert(0, grand_parent_directory)
from src.preprocessing.frame_generator import FrameGenerator
from src.preprocessing.face_extractor import FaceDetector
from src.preprocessing.normalization import Normalization
from src.preprocessing.face_alignment import FaceAlignment


class Training:
    """
    This class  is used to built, train and evaluate a CNN type of model
    """

    def __init__(self, config_file):
        self.__opt = Adam(learning_rate=config_file["train_params"]["learning_rate"])
        self.__es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
        self.__mc = ModelCheckpoint(config_file["model_dir"] + '/best_model.h5', monitor='val_accuracy',
                                    mode='max', verbose=1, save_best_only=True)
        self.__batch_size = config_file["train_params"]["batch_size"]
        self.__epochs = config_file["train_params"]["epochs"]
        self.__num_classes = config_file["train_params"]["num_classes"]
        self.__gray_color = config_file["train_params"]["gray_color"]
        self.__image_size = config_file["train_params"]["image_size"]
        self.__depth = config_file["train_params"]["num_frames_per_video"]
        self.__frames_per_second = config_file["train_params"]["frames_per_second"]
        self.__data_dir = config_file["data_dir"]
        self.__network_type = config_file["network_type"]
        self.__channel = 1 if self.__gray_color else 3

    def __build_mobile_net_v2(self):
        """
        this function build mobil_net model and returns the compiled model
        :return: mobile_net model
        """
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        emotions = Dense(self.__num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=emotions)
        for layer in base_model.layers:
            layer.trainable = True
        model.compile(optimizer=self.__opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_training_data(self):
        """
        Gets labeled training frames from videos in each actors directory
        inside the input directory.
        :return: list of frames, list of labels
        :rtype: ndarray
        """
        labels = []
        frames = []
        frame_gen = FrameGenerator(self.__frames_per_second)
        face_detect = FaceDetector()
        normalizer = Normalization( self.__gray_color, self.__image_size)
        face_align = FaceAlignment()
        actors = self.__read_input_data(self.__data_dir)
        for actor in actors:
            videos = self.__read_input_data(self.__data_dir + actor + '/')
            print("Current preprocessing : " + './videos/' + actor + '/')
            for filename in videos:
                video = cv2.VideoCapture(self.__data_dir + actor + '/' + filename)
                video_frames = frame_gen.get_equal_frames(video, self.__depth)
                if video_frames is not None or video_frames is not []:
                    face_frames = face_detect.get_frames(video_frames)
                    align_frames = face_align.get_frames(face_frames)
                    frames.append(normalizer.get_frames(align_frames))
                    labels.append(int(filename[6:8]) - 1)
        return np.array(frames), labels

    def __read_input_data(self, path):
        video_list = []
        for filename in os.listdir(path):
            video_list.append(filename)
        return video_list

    def split_data(self):
        """
        This function splits data into traning, testing and validation sets
        :return: 6 lists  with data for training, testing and validation.
        :rtype:ndarray
        """
        data, labels = self.get_training_data()
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2],
                             data.shape[3], self.__channel))
        labels = to_categorical(labels, self.__num_classes)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.10,
                                                            random_state=42, stratify=labels)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.111,
                                                          random_state=42, stratify=y_train)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def evaluate_model(self, model, x_test, y_test):
        """
        This function evaluates and displays the performance of the model on test data
        :param model: a trained model
        :type model: float
        :param x_test: array list of frames
        :type x_test: numpy array
        :param y_test: list of labels for frames
        :type y_test: numpy array
        """
        model_score = model.evaluate(x_test, y_test, verbose=1)
        print("Model accuracy: {}".format(model_score[1]))

    def __draw_model_accuracy_loss_graph(self, history):
        """
        This function draws two graphs: one for training and validation accuracy and
        one for training and validation loss
        :param history: an array that contains training accuracy and loss values of the total epochs
        """
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('model_accuracy_graph.png')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('model_loss_graph.png')

    def __convert_3d_to_2d_array(self, images, labels):
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
                if self.__gray_color:
                    x_new.append(images[i, j, :, :, 0])
                else:
                    x_new.append(images[i, j, :, :, :])
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                y_new.append(labels[i, :])
        if self.__gray_color:
            return np.repeat(np.array(x_new)[..., np.newaxis], 3, -1), np.array(y_new)
        else:
            return np.array(x_new), np.array(y_new)

    def train_model(self):
        """
        This function is used to train a cnn model
        """
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_data()
        x_train_new, y_train_new = self.__convert_3d_to_2d_array(x_train, y_train)
        x_val_new, y_val_new = self.__convert_3d_to_2d_array(x_val, y_val)
        x_test_new, y_test_new = self.__convert_3d_to_2d_array(x_test, y_test)
        print(x_train_new.shape, x_val_new.shape, x_test_new.shape)
        print(x_train.shape, x_val.shape, x_test.shape)
        model = self.__build_mobile_net_v2()
        print("========Training starts==========")
        model_history = model.fit(x_train_new, y_train_new, validation_data=(x_val_new, y_val_new),
                                  batch_size=self.__batch_size, epochs=self.__epochs, verbose=1,
                                  callbacks=[self.__mc, self.__es])
        print("=======evaluating model's test accuracy============")
        self.evaluate_model(model, x_test_new, y_test_new)
        print("======drawing model accuracy and loss=====")
        self.__draw_model_accuracy_loss_graph(model_history)
        print("=====Training finished=====")


if __name__ == '__main__':
    config = {}
    with open("training_config.yaml") as file:
        config = yaml.load(file)
        print(config)
    trainer = Training(config)
    trainer.train_model()

