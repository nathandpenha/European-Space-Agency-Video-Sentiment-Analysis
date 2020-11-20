from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.optimizers import Adam
from tensorflow import keras

opt = keras.optimizers.Adam(learning_rate =0.001)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('MyInceptionV3.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

img_height = 224
img_width = 224
batch_size = 16
nb_epochs = 1


class InceptionV3Trainer:

    def __init__(self, train_dir):
        self.train_dir_path = train_dir

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

    def prepare_training_data(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           validation_split=0.1)  # set validation split

        train_generator = train_datagen.flow_from_directory(
            self.train_dir_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')  # set as training data

        validation_generator = train_datagen.flow_from_directory(
            self.train_dir_path,  # same directory as training data
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation')  # set as validation data
        return train_generator, validation_generator

    def train_model(self):
        inception = InceptionV3Trainer('D:\\Projects\\esA\\us10\Dataset\\test_data\\frames-emotion_based_test_dataset')
        inception_model = inception.create_model()
        train_generator, validation_generator = inception.prepare_training_data()
        inception_model.compile(optimizer=opt, loss='categorical_crossentropy')
        print(inception_model.summary())
        inception_model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=nb_epochs, callbacks=[es, mc])
        inception_model.save('MyInceptionV3.h5')


if __name__ == '__main__':
    train_obj = InceptionV3Trainer("fake_path")
    train_obj.train_model()