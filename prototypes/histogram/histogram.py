"""
@Author Nathan DPenha, Akram Shokri
@Date 2 Nov 2020

This script get a live stream from camera and capture a frame from it in a loop.
This frame is fed to a trained model to estimate the emotional expressions.
The script will return a  percentage for eight sentiments which then will be shown as a histogram over the live feed.

"""
import cv2
import random

# The list of emotions displayed on the screen
EMOTIONS = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']


def live_feed():
    video_captor = cv2.VideoCapture(0)
    video_captor.set(cv2.CAP_PROP_FRAME_WIDTH, 1680)  # Set the width of the Frame
    video_captor.set(cv2.CAP_PROP_FRAME_HEIGHT, 1220)  # Set the height of the Frame
    result = None
    timer = 19  # Used to count the iteration
    while True:
        timer = timer + 1
        ret, frame = video_captor.read()  # Capture the image
        if timer % 20 == 0:
            cv2.imwrite('a.jpg', frame)  # Save the image to the disk
            result = [[]]  # Call the model and store the result in the result list
            # For loop for generating random data
            for i in range(0, 8):
                result[0].append(random.uniform(0.0, 1.0))  # Random values from 0 to 1
            print(result)
        if result is not None:
            cv2.rectangle(frame, (0, 0),
                          (150, 330),
                          (0, 0, 0), -1)  # Black background over text

            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 40 + 40), cv2.FONT_ITALIC, 0.75, (0, 255, 0), 1)  # Add Emotions
                cv2.rectangle(frame, (160, index * 40 + 20),
                              (160 + int(result[0][index] * 100 * 1.5), (index + 1) * 40 + 4),
                              (34, 139, 34), -1)  # Add bars for histogram
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


live_feed()
