import os
import sys
import pathlib
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
parser.add_argument('-quant', default=True, action='store_true', help='path to image data(train/val/test)')
parser.add_argument('-test', default=False, action='store_true', help='path to image data(train/val/test)')
args = parser.parse_args()

class ModelQuantization(object):

    def __init__(self, trained_keras_model_path):
        self.keras_model_path = trained_keras_model_path

    def quantize(self, qtype='float16'):
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(self.keras_model_path)
        tflite_model = converter.convert()
        tflite_models_dir = pathlib.Path("tf_lite_models")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        self.tflite_model_file = tflite_models_dir/"model.tflite"
        self.tflite_model_file.write_bytes(tflite_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_types = [tf.float16]
        tflite_fp16_model = converter.convert()
        self.tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_f16.tflite"
        self.tflite_model_fp16_file.write_bytes(tflite_fp16_model)

    # A helper function to evaluate the TF Lite model using "test" dataset.
    def evaluate_model(self, interpreter, generator):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # Run predictions on every image in the "test" dataset.
        predictions = []
        labels = []
        for i, gen_batch in enumerate(generator.test_generator):
            if i > generator.test_generator.samples: # samples in testset
                break
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            test_image, test_label = gen_batch
            #test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            pred = np.argmax(output()[0])
            predictions.append(pred)
            labels.append(test_label)

        # Compare prediction results with ground truth labels to calculate accuracy.
        accurate_count = 0
        for index in range(len(predictions)):
            if predictions[index] == labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(predictions)
        return accuracy

    def test(self, generator):
        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_file))
        interpreter.allocate_tensors()
        print('TFLiteModel:', self.evaluate_model(interpreter, generator))
        interpreter_fp16 = tf.lite.Interpreter(model_path=str(self.tflite_model_fp16_file))
        interpreter_fp16.allocate_tensors()
        print('TFLiteModelF16:', self.evaluate_model(interpreter_fp16, generator))

def main():
    input_data_path = args.data
    input_model_path = args.model
    input_model_width = args.width
    input_model_height = args.height

    if args.quant:
        if input_data_path is not None and os.path.exists(input_data_path):
            generator = ImageGenerator(input_data_path, (input_model_height, input_model_width),
                                                            16, 1, 'sparse')
        quant = ModelQuantization(input_model_path)
        quant.quantize()
        if generator:
            quant.test(generator)
    elif args.test:
        print ('Not implemented!!')
    else:
        print ('Unrecognized mode!!')

if __name__=='__main__':
    main()
