"""
Date: 23 November 2020
Authors: Hossein Mahdian, Vladimir Romashov

This is an output module script that is able to show histogram with emotion distribution
from camera stream.

The script is able to show histogram output in a separate window.
"""

import cv2


class GUIOutput:
    """
    This class manages histogram representation
    """

    __EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful']

    def draw_histogram(self, ir_result, image):
        """
        This function shows a histogram Inference's module results on Raspberry's camera stream
        :param ir_result: result from prediction
        :param image: captured image
        """
        if ir_result is not None:
            cv2.rectangle(image, (0, 0),
                           (150, int(len(self.__EMOTIONS) * 45)),
                           (0, 0, 0), -1)
            for index, emotion in enumerate(self.__EMOTIONS):
                cv2.putText(image, emotion, (10, index * 40 + 40), cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
                cv2.rectangle(image, (160, index * 40 + 20),
                               (160 + int(ir_result[index] * 100 * 1.5), (index + 1) * 40 + 4),
                               (0, 0, 0), -1)
            cv2.imshow("Frame", image)
