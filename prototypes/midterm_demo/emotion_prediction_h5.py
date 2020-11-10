# /usr/bin/python3
import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTIONS = ['Angry', 'Calm', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
image_size = (224, 224)
mobilenet_model = load_model("MobileNetV2.h5")

def live_feed(model_path):
    video_captor = cv2.VideoCapture(0)
    video_captor.set(cv2.CAP_PROP_FRAME_WIDTH, 1680)
    video_captor.set(cv2.CAP_PROP_FRAME_HEIGHT, 1220)
    result = None
    face_image = None
    timer = 19

    while True:
        timer = timer + 1
        ret, frame = video_captor.read()

        # number of frames that is used:
        if timer % 20 == 0:
           # cv2.imwrite('a.jpg', frame)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.3,
                minNeighbors=3)
            for (x, y, w, h) in faces:
                face_image = frame[y:y + h, x:x + w]

            if face_image is not None:
                # resize an image for being able to use in a NN
                resized_face_image = cv2.resize(face_image, image_size)
                resized_face_image = np.array(resized_face_image)
                # general normalization
                resized_face_image = resized_face_image / 255
                result = mobilenet_model.predict(resized_face_image.reshape(1,224,224,3))
                print(result)

        if result is not None:
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 25), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (150, index * 20 + 10), (150 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                          (34,139,34), -1)

        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

live_feed("model_path")