"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Hossein Mahdian h.mahdian@tue.nl, Vladimir Romashov v.romashov@tue.nl
@Description:
This is an output module script that is able to show histogram with emotion distribution
from camera stream.
The script is able to show histogram output in a separate window.
@Last modified date: 24-11-2020
"""

import cv2


class GUIOutput:
    """
    This class manages histogram representation
    """

    __EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful']

    def draw_histogram(self, ir_result, image):
        """shows a histogram Inference's module results on Raspberry's camera stream

        Args:
            ir_result: result from prediction
            image: captured image
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
