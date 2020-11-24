'''
Date: 12 November 2020
Authors: Samsom Beyene
This script extracts human faces from a list of images
and stores the result  in memory or to disk based on a flag from user input.
'''

import cv2
import sys
import os
import glob
import dlib
from .ipreprocessing import IPreprocessing


class FaceDetector(IPreprocessing):
    """
    This class is used for face detection
    """

    def __init__(self):
        self.__detector = dlib.get_frontal_face_detector()
        self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.__face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.__face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
        self.__face_cascade3 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.__face_cascade4 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def __extract_face(self, frames):
        """
        This function returns a list of detected human faces
        :param frames: list of images stored as type ndarray
        :type frames: list
        :return: a list of detected human faces
        :rtype: list
        """
        face_list = []
        for frame in frames:
            try:
                # detect human face using one of the five openCV libraries
                faces = self.__face_cascade.detectMultiScale(frame, 1.2, 5)
                if (len(faces) == 0 or len(faces) > 1):
                    faces = self.__face_cascade1.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
                    if (len(faces) == 0 or len(faces) > 1):
                        faces = self.__face_cascade2.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
                        if (len(faces) == 0 or len(faces) > 1):
                            faces = self.__face_cascade3.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
                            if (len(faces) == 0 or len(faces) > 1):
                                faces = self.__face_cascade4.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))

                if (len(faces) == 1):
                    # extract the detected face into new image
                    for (x, y, w, h) in faces:
                        face_image = frame[y:y + h, x:x + w]
                        face_list.append(face_image)
                else:
                    # detect human face using dlib
                    detections = self.__detector(frame, 1)

                    if len(detections) > 0:
                        d = detections[0]
                        left = d.left()
                        right = d.right()
                        top = d.top()
                        bottom = d.bottom()
                        face_image = frame[top:bottom, left:right]
                        face_list.append(face_image)
            except:
                if frame is None:
                    print("Frame is empty!")
                else:
                    print("can not detect human face: Either image does not have human face or image is corrupted")
        return face_list

    def get_frames(self, frame_list):
        """
        This function returns a list of detected human faces
        :param frame_list: list of images stored as type ndarray
        :type frame_list: list
        :return: a list of detected human faces
        :rtype: list
        """
        return self.__extract_face(frame_list)

    def save_frames(self, frame_dict, output_path):
        """
        This function saves the detected human faces to disk
        :param frame_dict: dictionary of frames with their filenames
        :type frame_dict: dictionary
        :param output_path: a path to save the detected human faces
        :type output_path: String
        """
        images_file_name = []
        images = []
        for key, value in frame_dict.items():
            images_file_name.append(key)
            images.append(value)
        face_images = self.__extract_face(images)
        for i, face_image in enumerate(face_images):
            cv2.imwrite(output_path + "/" + images_file_name[i], face_image)


if __name__ == '__main__':
    face_detector = FaceDetector()
    image_dict = {}
    image_list = []
    usage_message = """"Usage of face extractor:
                    > python face_extractor.py [input directory] [output directory] [flag]
                    """
    if len(sys.argv) != 4:
        print(usage_message)
        exit(0)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    save_flag = int(sys.argv[3])
    for image_file in sorted(glob.glob(input_path + '/*.jpg')):
        image = cv2.imread(image_file)
        if save_flag:
            image_dict[os.path.basename(image_file)] = image
        else:
            image_list.append(image)
    if save_flag:
        face_detector.save_frames(image_dict, output_path)
    else:
        faces = face_detector.get_frames(image_list)
        cv2.imshow("Sample extracted face", faces[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()