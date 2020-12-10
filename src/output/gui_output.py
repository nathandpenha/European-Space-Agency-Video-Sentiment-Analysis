"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Hossein Mahdian h.mahdian@tue.nl, Vladimir Romashov v.romashov@tue.nl
@Contributors: Tom Vrancken (t.j.g.m.vrancken@tue.nl)
@Description:
This is an output module script that is able to show histogram with emotion distribution
from camera stream.
The script is able to show histogram output in a separate window.
@Last modified date: 09-12-2020
"""

import cv2


class GUIOutput:
    """
    This class manages histogram representation
    """

    # List of emotions that are defined in 'training/config/data_preparation_config.yml' and are used
    # during ANN training.
    # IMPORTANT: The order of list items is important and must be in sync with the order in the
    #            aforementioned config file! Changing only one of the lists will result in incorrect
    #            classification results.
    __EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']

    # Colors used in the histogram widget:
    #  - red (negative emotions)
    #  - green (positive emotions)
    #  - white (neutral emotions)
    # IMPORTANT: The order of the items is important and corresponds to the items in __EMOTIONS.
    #            The item order for the __EMOTIONS and __COLORS lists must be kept in sync.
    __COLORS = [(255, 255, 255), (255, 255, 255), (34, 139, 34), (40, 20, 240), (40, 20, 240), (40, 20, 240),
                (40, 20, 240), (34, 139, 34)]

    def draw_histogram(self, prediction_result, image, enabled_emotions):
        """Draws a histogram with the inference results on top of a video frame

        Args:
            prediction_result: result from emotion prediction
            image: captured image
            enabled_emotions: a boolean list stating which emotions are enabled
        """
        if prediction_result is not None:
            num_enabled_emotions = 0
            for enabled in enabled_emotions:
                if enabled:
                    num_enabled_emotions += 1

            # Draw a black background for the widget
            cv2.rectangle(image, (0, 0),
                          (150, num_enabled_emotions * 40 + 20),
                          (0, 0, 0), -1)

            filtered_emotions = []
            filtered_colors = []
            for index, emotion in enumerate(self.__EMOTIONS):
                if enabled_emotions[index]:
                    filtered_emotions.append(emotion)
                    filtered_colors.append(self.__COLORS[index])

            i = 0
            for index, emotion in enumerate(filtered_emotions):
                # Draw histogram legend text
                cv2.putText(image, emotion, (10, i * 40 + 40), cv2.FONT_ITALIC, 0.75, filtered_colors[index], 1)
                # Draw histogram bar
                cv2.rectangle(image, (160, i * 40 + 20),
                              (160 + int(prediction_result[index] * 100 * 1.5), (i + 1) * 40 + 4),
                              filtered_colors[index], -1)
                i += 1

            cv2.imshow("Frame", image)

