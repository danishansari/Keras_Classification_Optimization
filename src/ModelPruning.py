import os
import sys
import numpy as np
from ImageGenerator import ImageGenerator

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.models import load_model

import argparse
parser = argparse.ArgumentParser(description='Model pruning')
parser.add_argument('-data', type=str, help='path to image data(train/val/test)')
parser.add_argument('-model', type=str, help='path to image data(train/val/test)')
parser.add_argument('-width', type=int, help='path to image data(train/val/test)')
parser.add_argument('-height', type=int, help='path to image data(train/val/test)')
parser.add_argument('-prune', default=True, action='store_true', help='path to image data(train/val/test)')
parser.add_argument('-test', default=False, action='store_true', help='path to image data(train/val/test)')
args = parser.parse_args()

class ModelPruning(object):

    def __init__(self, trained_keras_model_path):
        self.keras_model_path = trained_keras_model_path
        if os.path.exists(self.keras_model_path):
            self.model = tf.keras.models.load_model(self.keras_model_path)

    def prune_and_train(self, train_gen, test_gen, valid_gen, batch=16, epochs=3):
        keras_model_accuracy = 0.99
        _,  keras_model_accuracy= self.model.evaluate(valid_gen, verbose=0)
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        num_images = train_gen.samples 
        end_step = np.ceil(num_images / batch).astype(np.int32) * epochs

        # Define model for pruning.
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
                                                               }

        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)
        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        model_for_pruning.summary()
        #logdir = tempfile.mkdtemp()
        logdir = 'logs'
        callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
            ]

        model_for_pruning.fit(train_gen,
                  batch_size=batch,
                  epochs=epochs,
                  callbacks=callbacks)

        _, model_for_pruning_accuracy = model_for_pruning.evaluate(
                                               test_gen, verbose=0)

        
        print('Baseline test accuracy:', keras_model_accuracy)
        print('Pruned test accuracy:', model_for_pruning_accuracy)

        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        #_, pruned_keras_file = tempfile.mkstemp('.h5')
        pruned_keras_file = 'pruned_keras_model.h5'
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        print('Saved pruned Keras model to:', pruned_keras_file)


def main():
    input_data_path = args.data
    input_model_path = args.model
    input_model_width = args.width
    input_model_height = args.height

    if args.prune:
        gen = ImageGenerator(input_data_path, (input_model_height, input_model_width), 16, 1, 'sparse')
        gen_ = ImageGenerator(input_data_path, (input_model_height, input_model_width), 16, 1, 'categorical')

        prune = ModelPruning(input_model_path)
        prune.prune_and_train(gen.train_generator, gen.test_generator, gen_.valid_generator)
    elif args.test:
        print ('Not implemented!!')
    else:
        print ('Unrecognized mode!!')

if __name__=='__main__':
    main()
