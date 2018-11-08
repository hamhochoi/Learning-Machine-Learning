import os

import logging
import numpy as np
import cv2
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16

from utils.common import env_utils, img_utils
from utils.common import logging_utils
from utils.common import shared_params
from utils.common.config_loader import ConfigLoader


class VGGBasedClassifier(object):
    FIELD_DATA_DIR = 'data_dir'
    FIELD_WEIGHTS_PATH = 'weights_path'
    FIELD_MODEL_PATH = 'model_path'
    FIELD_IMG_WIDTH = 'img_width'
    FIELD_IMG_HEIGHT = 'img_height'
    FIELD_IMG_DEPTH = 'img_depth'
    FIELD_BATCH_SIZE = 'batch_size'
    FIELD_EPOCHS = 'epochs'
    FIELD_FINE_TUNING = 'fine_tuning'
    FIELD_FINE_TUNING_EPOCHS = 'fine_tuning_epochs'
    FIELD_WARM_START = 'warm_start'

    def __init__(self, data_dir='data', weights_path='models/weights.h5', model_path='models/model.h5',
                 img_width=448, img_height=448, img_depth=3, rescale=1./255,
                 batch_size=32, epochs=30, fine_tuning=True, fine_tuning_epochs=50, warm_start=False):
        super(VGGBasedClassifier, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.rescale = rescale
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tuning = fine_tuning
        self.warm_start = warm_start
        self.fine_tuning_epochs = fine_tuning_epochs
        self.weights_path = weights_path
        self.model_path = model_path
        self.data_dir = data_dir
        self.conv_base = None
        self.model = None

    def build_model(self, compile_model=False, load_weigts=False, weights_path=None, lr=5e-5):
        self.conv_base = VGG16(include_top=False, weights='imagenet',
                               input_shape=(self.img_width, self.img_height, self.img_depth))
        self.conv_base.trainable = False
        self.model = models.Sequential()
        self.model.add(self.conv_base)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.summary()
        if compile_model:
            weights_path = weights_path or self.weights_path
            weights_path = env_utils.get_absolute_path(weights_path)
            if load_weigts and weights_path:
                self.model.load_weights(weights_path)
            self.model.compile(RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc'])

    def load_model(self, model_path=None):
        model_path = model_path or self.model_path
        model_path = env_utils.get_absolute_path(model_path)
        self.model = models.load_model(model_path)

    def save_model(self, model_path=None):
        model_path = model_path or self.model_path
        model_path = env_utils.get_absolute_path(model_path)
        self.model.save(model_path)

    def train(self, train_dir=None, val_dir=None,
              batch_size=None, epochs=None,
              warm_start=None, lr=5e-5,
              fine_tuning=None, fine_tuning_epochs=None,
              weights_path=None, model_path=None,
              show_history=True):
        weights_path = weights_path or self.weights_path
        weights_path = env_utils.get_absolute_path(weights_path)
        model_path = model_path or self.model_path
        model_path = env_utils.get_absolute_path(model_path)
        train_dir = train_dir or os.path.join(self.data_dir, 'train')
        train_dir = env_utils.get_absolute_path(train_dir)
        val_dir = val_dir or os.path.join(self.data_dir, 'validation')
        val_dir = env_utils.get_absolute_path(val_dir)
        batch_size = batch_size or self.batch_size
        epochs = epochs or self.epochs
        if fine_tuning is None:
            fine_tuning = self.fine_tuning
        fine_tuning_epochs = fine_tuning_epochs or self.fine_tuning_epochs
        if warm_start is None:
            warm_start = self.warm_start

        self.build_model(compile_model=True, load_weigts=warm_start, weights_path=weights_path, lr=lr)
        train_idg = image.ImageDataGenerator(rotation_range=30,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True,
                                             rescale=self.rescale)
        train_generator = train_idg.flow_from_directory(train_dir,
                                                        target_size=(self.img_width, self.img_height),
                                                        class_mode='binary',
                                                        batch_size=batch_size or 32)
        val_idg = image.ImageDataGenerator(rescale=self.rescale)
        val_generator = val_idg.flow_from_directory(val_dir,
                                                    target_size=(self.img_width, self.img_height),
                                                    class_mode='binary',
                                                    batch_size=batch_size or 32)
        train_steps = train_generator.n // batch_size + int(train_generator.n % batch_size != 0)
        val_steps = val_generator.n // batch_size + int(val_generator.n % batch_size != 0)
        history = self.model.fit_generator(train_generator,
                                           steps_per_epoch=train_steps,
                                           epochs=epochs,
                                           callbacks=[ModelCheckpoint(weights_path,
                                                                      save_best_only=True,
                                                                      save_weights_only=True)],
                                           validation_data=val_generator,
                                           validation_steps=val_steps)
        self.model.load_weights(weights_path)

        if show_history:
            logging.getLogger(__name__).info('Training process')
            self.show_history(history)

        if fine_tuning:
            logging.getLogger(__name__).info('Unfreezing conv_base')
            self.conv_base.trainable = True
            set_trainable = False
            for layer in self.conv_base.layers:
                if layer.name == 'block5_conv1':
                    set_trainable = True
                layer.trainable = set_trainable
            self.model.compile(optimizer=RMSprop(lr=1e-5), loss='binary_crossentropy', metrics=['acc'])
            train_steps = train_generator.n // batch_size + int(train_generator.n % batch_size != 0)
            val_steps = val_generator.n // batch_size + int(val_generator.n % batch_size != 0)
            history = self.model.fit_generator(train_generator,
                                               steps_per_epoch=train_steps,
                                               epochs=fine_tuning_epochs,
                                               callbacks=[ModelCheckpoint(weights_path,
                                                                          save_best_only=True,
                                                                          save_weights_only=True)],
                                               validation_data=val_generator,
                                               validation_steps=val_steps)
            self.model.load_weights(weights_path)
            logging.getLogger(__name__).info('Re-freezing conv_base')
            for layer in self.conv_base.layers:
                layer.trainable = False
            self.conv_base.trainable = False
            self.model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc'])
            if show_history:
                logging.getLogger(__name__).info('Fine-tuning process')
                self.show_history(history)
        if weights_path:
            self.model.save_weights(weights_path)
        if model_path:
            self.model.save(model_path)

    @staticmethod
    def show_history(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = history.epoch
        for epoch, a, va, l, vl in zip(epochs, acc, val_acc, loss, val_loss):
            logging.getLogger(__name__).info('epoch: %s, acc: %.3f, val_acc: %.3f, loss: %.3f, val_loss: %.3f'
                                             % (epoch, a, va, l, vl))
        logging.getLogger(__name__).info(50 * '==')

    def test(self, test_dir=None, weights_path=None, model_path=None, limited_memory=False):
        if self.model is None:
            weights_path = weights_path or self.weights_path
            weights_path = env_utils.get_absolute_path(weights_path)
            model_path = model_path or self.model_path
            model_path = env_utils.get_absolute_path(model_path)
            if weights_path and os.path.exists(weights_path):
                self.build_model(compile_model=True)
                self.model.load_weights(weights_path)
            elif model_path and os.path.exists(model_path):
                self.load_model(model_path)
        test_dir = test_dir or os.path.join(self.data_dir, 'test')
        test_dir = env_utils.get_absolute_path(test_dir)

        positive_dir = os.path.join(test_dir, 'positive')
        negative_dir = os.path.join(test_dir, 'negative')
        tp, fp = self.evaluate(positive_dir, 1)
        tn, fn = self.evaluate(negative_dir, 0)
        logging.getLogger(__name__).info('True positive: %d, false positive: %d, total: %d' % (tp, fp, tp + fp))
        logging.getLogger(__name__).info('True negative: %d, false negative: %d, total: %d' % (tn, fn, tn + fn))
        logging.getLogger(__name__).info('Accuracy: %.4f' % (float(tp + tn) / (tp + fp + tn + fn),))

    def evaluate(self, test_dir, label, limited_memory=False, verbose=1):
        logging.getLogger(__name__).info('Evaluate directory %s' % test_dir)
        test_dir = env_utils.get_absolute_path(test_dir)
        correct_ones = 0
        incorrect_ones = 0
        for file_name in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file_name)
            prediction = self.chunk_predict(file_path, limited_memory)
            correct = round(prediction) == label
            if correct:
                correct_ones += 1
            else:
                incorrect_ones += 1
            if verbose:
                logging.getLogger(__name__).info('%s %s %s %s' % (file_path, label, prediction, correct))
        return correct_ones, incorrect_ones

    def predict(self, file_path):
        if self.model is None:
            self.build_model(compile_model=True, load_weigts=True)
        img = cv2.imread(file_path)
        height, width, depth = img.shape
        if height < width:
            img1 = img[:, :height, :]
            img2 = img[:, width - height:, :]
        else:
            img1 = img[:width, :, :]
            img2 = img[height - width:, :, :]
        img0 = cv2.resize(img, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC) * self.rescale
        img1 = cv2.resize(img1, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC) * self.rescale
        img2 = cv2.resize(img2, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC) * self.rescale
        x = np.array([img0, img1, img2])
        predictions = self.model.predict(x)
        prediction = np.max(predictions)
        return prediction

    def chunk_predict(self, file_path, max_chunks=16, limited_memory=True):
        if self.model is None:
            self.build_model(compile_model=True, load_weigts=True)
        img = cv2.imread(file_path)
        imgs = img_utils.chunk(img, self.img_width, self.img_height, max_chunks)
        img_resize = cv2.resize(img, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
        imgs.append(img_resize)
        x = np.array(imgs) * self.rescale
        if limited_memory:
            predictions = []
            for img_i in x:
                prediction = self.model.predict(np.array([img_i]))
                predictions.extend(prediction)
        else:
            predictions = self.model.predict(x)
        prediction = np.max(predictions)
        return prediction

    @staticmethod
    def load_config(config_path=shared_params.config_path, config_section=None):
        config = {}
        config_path = env_utils.get_absolute_path(config_path)
        if os.path.exists(config_path):
            loader = ConfigLoader(config_path)
            config_section = config_section or VGGBasedClassifier.__name__
            config[VGGBasedClassifier.FIELD_DATA_DIR] = loader.get(config_section, VGGBasedClassifier.FIELD_DATA_DIR)
            config[VGGBasedClassifier.FIELD_WEIGHTS_PATH] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_WEIGHTS_PATH)
            config[VGGBasedClassifier.FIELD_MODEL_PATH] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_MODEL_PATH)
            config[VGGBasedClassifier.FIELD_IMG_WIDTH] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_IMG_WIDTH, map_func=int)
            config[VGGBasedClassifier.FIELD_IMG_HEIGHT] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_IMG_HEIGHT, map_func=int)
            config[VGGBasedClassifier.FIELD_IMG_DEPTH] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_IMG_DEPTH, map_func=int)
            config[VGGBasedClassifier.FIELD_BATCH_SIZE] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_BATCH_SIZE, map_func=int)
            config[VGGBasedClassifier.FIELD_EPOCHS] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_EPOCHS, map_func=int)
            config[VGGBasedClassifier.FIELD_FINE_TUNING] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_FINE_TUNING, map_func=ConfigLoader.bool_map)
            config[VGGBasedClassifier.FIELD_WARM_START] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_WARM_START, map_func=ConfigLoader.bool_map)
            config[VGGBasedClassifier.FIELD_FINE_TUNING_EPOCHS] = \
                loader.get(config_section, VGGBasedClassifier.FIELD_FINE_TUNING_EPOCHS, map_func=int)
        for key in config.keys():
            if config[key] is None:
                del config[key]
        return config


if __name__ == '__main__':
    logging_utils.setup_logging()
    classifier = VGGBasedClassifier(**VGGBasedClassifier.load_config())
    # classifier.build_model(compile_model=True, load_weigts=True)
    classifier.load_model('weights/lastest_model.h5')
    test_dir = 'data/error/boy/error/'  # path from project dir or absolute path
    # test_dir = env_utils.get_absolute_path(test_dir)
    for file_name in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file_name)
        prob = classifier.chunk_predict(file_path, 16)
        print (file_name, prob)
