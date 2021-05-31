import os
import cv2
import random
import time
import numpy as np
from collections import Counter
from squeezenet import SqueezeNet
from ImageGenerator import ImageGenerator

import tensorflow as tf
import keras
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#tf.keras.backend.set_session(tf.Session(config=config));

import argparse
parser = argparse.ArgumentParser(description='Model pruning')
parser.add_argument('-data', type=str, help='path to image data(train/val/test)')
parser.add_argument('-model', default='models', type=str, help='path to image data(train/val/test)')
parser.add_argument('-width', type=int, help='path to image data(train/val/test)')
parser.add_argument('-height', type=int, help='path to image data(train/val/test)')
parser.add_argument('-batch', type=int, default=16, help='path to image data(train/val/test)')
parser.add_argument('-epoch', type=int, default=10, help='path to image data(train/val/test)')
parser.add_argument('-lr', type=float, default=0.001, help='path to image data(train/val/test)')
parser.add_argument('-train', default=False, action='store_true', help='path to image data(train/val/test)')
parser.add_argument('-eval', default=False, action='store_true', help='path to image data(train/val/test)')
parser.add_argument('-test', default=False, action='store_true', help='path to image data(train/val/test)')
parser.add_argument('-inference', default=False, action='store_true', help='path to image data(train/val/test)')
args = parser.parse_args()


class Classification(object):

        def __init__(self, data, arch, batch, size, classes):
            self.data_path  = data
            self.model_arch = arch
            self.batch = batch
            self.n_classes = classes
            self.size = size
            self.gen = ImageGenerator(self.data_path, size, batch, 1, 'categorical')
            self.classes = self.gen.train_generator.class_indices
            self.input_shape = size + tuple([3])
            self.model = SqueezeNet(include_top=False,
                    weights='imagenet',
                    input_shape=self.input_shape,
                    pooling='avg',
                    classes=self.n_classes)
            x = Dense(self.n_classes, activation='softmax', name='predictions')(self.model.layers[-1].output)
            self.model = keras.Model(input=self.model.inputs, output=x)
            counter = Counter(self.gen.train_generator.classes)
            max_val = float(max(counter.values()))
            self.class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
            print ('Classes:', self.classes)
            print ('Class weights:', self.class_weights)

        def load_model(self, model_path):
            self.model_path = model_path
            self.model = tf.keras.models.load_model(self.model_path)
            if 'prun' in model_path:
                self.model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        def smoothing_loss(self, y_true, y_pred):
            return tf.losses.sigmoid_cross_entropy(y_true, y_pred, label_smoothing=0.9)

        def train(self, lr_rate=0.001, n_epochs=10):
            checkpoint_callback = ModelCheckpoint(
                    filepath='models/checkpoint_{epoch:02d}_{val_acc:.2f}.h5',
                    period=1,
                    monitor='val_acc')

            #self.model.compile(loss=keras.losses.categorical_crossentropy,
            self.model.compile(loss=self.smoothing_loss,
                    optimizer=keras.optimizers.Adam(lr_rate),
                    metrics=['acc'])

            self.model.fit_generator(
                    self.gen.train_generator,
                    steps_per_epoch=self.gen.train_generator.samples // self.batch,
                    epochs=n_epochs,
                    validation_data=self.gen.valid_generator,
                    validation_steps=self.gen.valid_generator.samples // self.batch,
                    callbacks=[checkpoint_callback],
                    class_weight=self.class_weights)

        def evaluate(self):
            evals = self.model.evaluate(self.gen.valid_generator, verbose=0)
            print ('Model evaluate:', evals)

        def test(self):
            correct_prection = [0 for i in range(self.n_classes)]
            total_samples = [0 for i in range(self.n_classes)]
            for i, batches in enumerate(self.gen.test_generator):
                if i > 2003:
                    break
                image, label = batches
                pred = self.model.predict(image)[0]
                pred = np.argmax(pred)
                if pred == np.argmax(label):
                    correct_prection[pred] += 1
                total_samples[pred] += 1
            acc = []
            for i, c in enumerate(correct_prection):
                acc.append(c/total_samples[i])
            print ('Test accuracy:', acc)
            return acc

        def inference(self):
            test_path = '/test'
            if not os.path.exists(self.data_path+test_path):
                test_path = '/valid'
            correct_prection = [0 for i in range(self.n_classes)]
            total_samples = [0 for i in range(self.n_classes)]
            show = False
            mtime = 0.0
            stime = time.time()
            for i, classes in enumerate(os.listdir(self.data_path+test_path)):
                for files in os.listdir(self.data_path+test_path+'/'+classes):
                    filepath = os.path.join(self.data_path+test_path+'/'+classes, files)
                    if os.path.exists(filepath):
                        src = cv2.imread(filepath)
                        if src is not None:
                            img = src[...,::-1] # BGR->RGB
                            img = cv2.resize(img, (self.size[1], self.size[0]))
                            img = img.astype(np.float32)/255.0
                            img = np.expand_dims(img, axis=0)
                            t1 = time.time()
                            pred = self.model.predict(img)[0]
                            t2 = time.time()
                            mtime += t2-t1
                            pred = np.argmax(pred)
                            if pred == self.classes[classes]:
                                correct_prection[i] += 1
                            total_samples[i] += 1
                        else:
                            print ('Could not load test image!!')
                    else:
                        print ('Could not find test image!!')
            acc = []
            for i, c in enumerate(correct_prection):
                acc.append(c/total_samples[i])
            print ('Test accuracy:', acc, sum(acc)/4.0, 'Model time:', mtime, mtime/sum(total_samples), 'Total time = ', time.time()-stime)
            return acc


def main():
    input_data_path = args.data
    input_model_width = args.width
    input_model_height = args.height
    batch_size = args.batch
    num_classes = 4
    print ('args:', args.train, args.test)
    if args.train:
        n_epochs = args.epoch
        classifier = Classification(input_data_path, 'squeezenet', batch_size, (input_model_height, input_model_width), num_classes)
        classifier.train(args.lr, args.epoch)
    elif args.eval:
        model_path = args.model
        classifier = Classification(input_data_path, 'squeezenet', batch_size, (input_model_height, input_model_width), num_classes)
        classifier.load_model(model_path)
        classifier.evaluate()
    elif args.test:
        model_path = args.model
        classifier = Classification(input_data_path, 'squeezenet', batch_size, (input_model_height, input_model_width), num_classes)
        classifier.load_model(model_path)
        classifier.test()
    elif args.inference:
        model_path = args.model
        classifier = Classification(input_data_path, 'squeezenet', batch_size, (input_model_height, input_model_width), num_classes)
        classifier.load_model(model_path)
        classifier.inference()
    else:
        print ('Unrecognized mode!!')


if __name__=='__main__':
    main()
