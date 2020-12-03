"""
Date: 23 November 2020
Authors: Hossein Mahdian, Vladimir Romashov
Contributors: Akram Shokri (a.shokri@tue.nl)

This is an output module script that is able to show histogram with emotion distribution
from camera stream.

The script is able to show histogram output in a separate window.
"""

import cv2
import random


class GUIOutput:
    """
    This class manages histogram representation
    """

    # List of emotions to use according to training/config/data_preparation_config.yml. Do not change order of the items
    __EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']

    # Colors used in the histogram: red (negative emotions), green (positive emotions), white (neutral emotions).
    # Don't change the order of items
    __COLORS = [(255, 255, 255), (255, 255, 255), (34, 139, 34), (40, 20, 240), (40, 20, 240), (40, 20, 240),
                (40, 20, 240), (34, 139, 34)]

    def draw_histogram(self, ir_result, image, enabled_emotions_status):
        """
        This function shows a histogram Inference's module results on Raspberry's camera stream
        :param ir_result: result from prediction
        :param image: captured image
        :param enabled_emotions_status: a list containing True or False to show which emotions is enabled
        """
        if ir_result is not None:
            num_of_enabled_emotions = 0
            for enabled in enabled_emotions_status:
                if enabled:
                    num_of_enabled_emotions += 1

            cv2.rectangle(image, (0, 0),
                          (150, num_of_enabled_emotions * 40 + 20),
                          (0, 0, 0), -1)  # Black background over text

            i = 0
            for index, emotion in enumerate(self.__EMOTIONS):
                if (enabled_emotions_status[index]):
                    cv2.putText(image, emotion, (10, i * 40 + 40), cv2.FONT_ITALIC, 0.75, self.__COLORS[index],
                                1)  # Add Emotions
                    cv2.rectangle(image, (160, i * 40 + 20),
                                  (160 + int(ir_result[index] * 100 * 1.5), (i + 1) * 40 + 4),
                                  self.__COLORS[index], -1)  # Add bars for histogram
                    i += 1

            cv2.imshow("Frame", image)
