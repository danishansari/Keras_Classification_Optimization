import os
from keras.preprocessing.image import ImageDataGenerator

class ImageGenerator(object):

    def __init__(self, data_path, size, batch_train, batch_test, class_mode='categorical'):
        
        train_gen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

        valid_gen = ImageDataGenerator(
                rescale=1./255)

        if os.path.exists(data_path+'/train'):
            self.train_generator = train_gen.flow_from_directory(data_path+'/train',
                target_size=size,
                batch_size=batch_train,
                class_mode=class_mode)

        if os.path.exists(data_path+'/train'):
            self.valid_generator = train_gen.flow_from_directory(data_path+'/valid',
                target_size=size,
                batch_size=batch_test,
                class_mode=class_mode)

        if os.path.exists(data_path+'/test'):
            self.test_generator = train_gen.flow_from_directory(data_path+'/test',
                target_size=size,
                batch_size=batch_test,
                class_mode=class_mode)
        else:
            self.test_generator = self.valid_generator

