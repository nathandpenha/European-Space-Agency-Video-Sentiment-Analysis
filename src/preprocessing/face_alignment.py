"""
Date : 19 November 2020
Authors: Akram Shokri
"""
import cv2
import dlib
import os
from ipreprocessing import IPreprocessing
from frame_generator import FrameGenerator


class FaceAlignment(IPreprocessing):
    """
    This class is for aligning faces inside frames so as the eyes are in one line and both parallel to the x-axis.
    """
    def get_frames(self, frame_list):
        """
        This method accepts a frame as input and align the face of the subject in that frame.
        It returns the result as a frame.
        :param frame_list: list of objects of type OpenCV frame
        """
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            "../../models/shape_predictor_68_face_landmarks.dat")

        aligned_faces = []
        for frame in frame_list:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector(gray, 1)
            if len(detections) > 0 and frame is not None:
                detected_face = detections[0]
                img_shape = predictor(frame, detected_face)
                aligned_face = dlib.get_face_chip(frame, img_shape, size=frame.shape[0], padding=0.00)
                aligned_faces.append(aligned_face)
        return aligned_faces

    def save_frames(self, frame_dict, output_path):
        """
        This method accepts a dictionary and an input path. The dictionary contains filename as key
        and the corresponding frame as value. It aligns the face of the subject in all frames and save the aligned
        frames as a .jpg file in the output_path.
        :param frame_dict: A dictionary containing filename as the key and object of type OpenCV frame as values
        :param output_path: A string containing the path for saving the aligned frames
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        frames = []
        images_file_name = []
        for filename, frame in frame_dict.items():
            images_file_name.append(filename)
            frames.append(frame)

        aligned_frames = self.get_frames(frames)
        for i in range(0, len(aligned_frames)):
            if aligned_frames[i] is not None:
                cv2.imwrite(output_path + "/" + images_file_name[i].replace(".jpg","") + "-" + str(i) + ".jpg",
                            aligned_frames[i])


def main():
    output_path = "./output/"
    cap = cv2.VideoCapture(
        "./1.mp4")
    fg = FrameGenerator(6)
    fg.save_frames({"video": cap}, output_path)

    test_frames = {}
    for filename in os.listdir(output_path):
        if ".jpg" in filename:
            image = cv2.imread(output_path + filename)
            test_frames[filename] = image

    fn = FaceAlignment()
    fn.save_frames(test_frames, output_path + "aligned/")


if __name__ == "__main__":
    main()
