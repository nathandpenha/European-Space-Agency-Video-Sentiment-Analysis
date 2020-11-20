from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

img_height = 224
img_width = 224
batch_size = 16

class ModelEvaluater:

    def __init__(self, test_dir):
        self.test_dir_path = test_dir

    def load_model(self):
        return load_model(self.test_dir_path)

    def create_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        emotions = Dense(8, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=emotions)
        for layer in base_model.layers:
            layer.trainable = True
        return model

    def prepare_test_data(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    )  # set validation split

        train_generator = train_datagen.flow_from_directory(
            self.test_dir_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')  # set as training data

        validation_generator = train_datagen.flow_from_directory(
            self.test_dir_path,  # same directory as training data
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation')  # set as validation data
        return train_generator, validation_generator

    def prepare_test_dataset(self):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            directory=r'D:\\Projects\\esA\\us10\\Dataset\\test_data\\frames-emotion_based_test_dataset',
            target_size=(img_height, img_width),
            batch_size = 1,
            class_mode='categorical',
            # class_mode = None,
            shuffle = False
        )
        return test_generator

    def test_model(self):
        model = self.create_model()
        # model.load_weights()
        inception_model = load_model(self.test_dir_path)
        test_generator = self.prepare_test_dataset()
        loss = inception_model.evaluate_generator(test_generator, steps=24)
        print('loss : '+loss)

if __name__ == '__main__':
    test_obj = ModelEvaluater('D:\\Projects\\esA\\us10\\Dataset\\InceptionV3.h5')
    test_obj.test_model()
