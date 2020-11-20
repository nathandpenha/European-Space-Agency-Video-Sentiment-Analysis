import os
from face_extractor import FaceExtractor

input_root = 'D:\\Projects\\esA\\us10\\Dataset\\frames-emotion_based_dataset'
output_root = 'D:\\Projects\\esA\\us10\\Dataset\\frames-only_face_dataset'


def extract_face():
    emotions = ['01', '02', '03', '04', '05', '06', '07', '08']
    for emotion in emotions:
        output_path = output_root + '\\' + emotion
        input_path = input_root + '\\' + emotion
        print(output_path)
        print(input_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        face_extractor_obj = FaceExtractor(input_path, output_path, ".jpg")
        face_extractor_obj.face_extractor()


extract_face()
