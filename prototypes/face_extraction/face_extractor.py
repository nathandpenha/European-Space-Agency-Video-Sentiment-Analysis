"""
Copyright (c) 2020 TU/e - PDEng Software Technology C2019. All rights reserved.
@Author: Samsom Beyene s.t.beyene@tue.nl
@Last modified date: 30-10-2020
"""

import cv2
import glob
import os
from skimage.color import rgb2gray
from skimage.io import imread

class FaceExtractor:
    def __init__(self, input_path, output_path, image_format):

        self.input_folder = input_path
        self.output_folder = output_path
        self.format_type = image_format

    def face_extractor(self, save_flag = True):
        """
        this function detects a face from an input image and saves the extracted
        face image into a given output directory or store into a variable list according to the
        function argument value
        """
        face_list = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for image_file in sorted(glob.glob(self.input_folder + '/*.' + self.format_type)):
            try:
                image = cv2.imread(image_file)
            except imageError:
                print("Can not read image: Either the file is corrupted or it is not in the proper format")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )
            for (x,y,w,h) in faces:
                face_image = image[y:y+h, x:x+w]
                if save_flag:
                    cv2.imwrite(self.output_folder + "/"+ os.path.basename(image_file), face_image)
                else:
                    face_list.append(face_image)
        if not save_flag:
            return face_list

if __name__ == '__main__':
    face_extractor_obj = FaceExtractor("../input_folder", "../output_folder", "jpg" )
    face_extractor_obj.face_extractor()
