# /usr/bin/python3
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import load_model
import numpy as np

EMOTIONS = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
image_size = (48, 48)
sam_model = load_model("visual_modleperclip50.h5")
base_model_vgg16 = VGG16(include_top = False,input_shape =(48,48,3),pooling = 'avg', weights = 'imagenet')

def normalization_of_data(vdat):
    for j, it in enumerate(vdat):
        data = cv2.cvtColor(vdat[j], cv2.COLOR_BGR2GRAY)
        data = np.array(data, 'float32')
        data = data / 255
        vdat[j] = data
    return vdat

def change_input_into_3D(x_input, size):
    vgg_input = np.empty([size, 48, 48, 3])
    for index, item in enumerate(vgg_input):
        item[:, :, 0] = x_input[index]
        item[:, :, 1] = x_input[index]
        item[:, :, 2] = x_input[index]
    return vgg_input

def vgg16_feature_extraction(vdata):
    vdata = np.array(vdata)
    test3d= np.array(change_input_into_3D(vdata, int(len(vdata))))
    test_feature= base_model_vgg16.predict(test3d)
    return test_feature

def live_feed(model_path):
    video_captor = cv2.VideoCapture(0)
    result = None
    timer = 19
    face_image = None
    input_images = []
    while True:
        timer = timer + 1
        ret, frame = video_captor.read()

        # if detected_face is not None
        # number of frames that is used for the NN
        if timer % 2 == 0:
            #cv2.imwrite('a.jpg', frame)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.3,
                minNeighbors=3
            )
            cv2.imshow('face', frame)
            for (x, y, w, h) in faces:
                face_image = frame[y:y + h, x:x + w]
            if face_image is not None:
                resized_face_image = cv2.resize(face_image, image_size)
                resized_face_image=cv2.cvtColor(resized_face_image, cv2.COLOR_BGR2GRAY)
                resized_face_image = np.array(resized_face_image)
                resized_face_image = resized_face_image/255
                input_images.append(resized_face_image)
                print("frame{} taken:".format((len(input_images))))

                if len(input_images) < 45:
                    continue
                print(np.array(input_images).shape)
                x_test = vgg16_feature_extraction(input_images)
                x_test = np.reshape(x_test, (1, 45, 512))

                result = sam_model.predict(x_test)
                Emotion = {0: "Angry",
                           1: "Disgust",
                           2: "Fear",
                           3: "Happy",
                           4: "Netural",
                           5: "Sad",
                           6: "Surprise"}
                label = result.argmax()
                print("this is the probabilities of the emotions")
                for i in range(7):
                    print(Emotion[i], ':  ', result[0][i])
                print()
                print("the emotion is:" + str(Emotion[label]) + "----with prob of:" + str((result[0][label])))

        if result is not None:
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 25), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (150, index * 20 + 10), (150 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                          (34,139,34), -1)
        cv2.imshow('face', frame)
        if len(input_images)>=45:
            input_images =[]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

live_feed("model_path")