"""
Copyright (c) 2020 TU/e - PDEng Software Technology c2019. All rights reserved.
@Author: Samsom Beyene s.t.beyene@tue.nl
@Description: This script builds different type of (3d)cnn model architectures
@Last modified date: 27-11-2020
"""

from abc import ABC, abstractmethod
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Conv3D, Dropout, Flatten, MaxPooling3D
from keras.models import Model, Sequential


class IModelBuilder(ABC):
    """
    This is an interface class to build different types of cnn architectures
    """
    @abstractmethod
    def build_model(self, num_classes, num_filters, kernel_size, input_shape, depth):
        """abstract function

        Args:
            num_classes: number of classes
            num_filters: convolution filters
            kernel_size: filters
            input_shape: input shape of the model
            depth: sequence of frames needed for 3d cnn
        """
        pass


class MobileNet(IModelBuilder):
    """
    This class is used to create a MobilenetV2 cnn model type
    """
    def build_model(self, num_classes, num_filters, kernel_size, input_shape, depth):
        """builds mobil_net model and returns the  model

        Returns:
            mobile_net model
        """
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        emotions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=emotions)
        for layer in base_model.layers:
            layer.trainable = True
        return model


class ThreeDCNNV1(IModelBuilder):
    """
    This class is used to create a 3D CNN model type with Three convolution layers
    """
    def build_model(self, num_classes, num_filters, kernel_size, input_shape, depth):
        """builds a 3D CNN model and returns the  model

        Returns:
            3d_model
        """
        model = Sequential()
        _3d_input_shape = (input_shape[0], input_shape[1], depth, input_shape[2])
        kernel_size = (kernel_size[0], kernel_size[1], kernel_size[2])
        model.add(Conv3D(num_filters[0], kernel_size=kernel_size, input_shape=_3d_input_shape
                         , padding='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(num_filters[0], kernel_size=kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
        model.add(Conv3D(num_filters[1], kernel_size=kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(num_filters[1], kernel_size=kernel_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(num_classes, activation='softmax'))
        return model
