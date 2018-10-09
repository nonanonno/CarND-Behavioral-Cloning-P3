import numpy as np
import sys
import matplotlib.pyplot as plt
import csv



def LoadTrainingData(data_list, add_flip):
    """
    Load training data

    parameters
    ----------
    data_list : list
        training data prefix list to load
    add_flip : bool
        Add flipped image to training data
    """

    images = []
    steering_angles = []
    
    for prefix in data_list:
        image_prefix = prefix+'IMG/'
        lines = [line for line in csv.reader(open(prefix+'driving_log.csv'))]

        for line in lines:
            source_path = line[0]
            filename = source_path.split('\\')[-1][6:]
            
            # Use all of center images and left images, right images
            # For left image, correction value 0.2 is added to steering angle
            # For right image, correction value 0.2 is subtracted to steering angle
            correction = 0.2
            steering_center = float(line[3])
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            
            img_center = plt.imread(image_prefix+'center'+filename)
            img_left = plt.imread(image_prefix+'left'+filename)
            img_right = plt.imread(image_prefix+'right'+filename)
            
            images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])
            
            # If add_flip is true, training data contain flipped images.
            if add_flip:
                images.extend([
                    np.fliplr(img_center),
                    np.fliplr(img_left),
                    np.fliplr(img_right),
                ])

                steering_angles.extend([
                    steering_center*-1,
                    steering_left*-1,
                    steering_right*-1
                ])
    
    return np.array(images), np.array(steering_angles)


def SetupKeras():
    """
    Set keras to use 70% of GPU memory to run other programs that use GPU.
    """
    import tensorflow as tf
    from keras import backend as K
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    K.set_session(sess)


def CreateModel(shape):
    """
    Create convolution neural network model
    The model is based on nvidia's model
    https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    """
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Activation, Dropout, Cropping2D

    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(24,(5,5),activation='relu'))

    model.add(MaxPooling2D(strides=(2,2)))

    model.add(Conv2D(36,(5,5),activation='relu'))
    model.add(MaxPooling2D(strides=(2,2)))

    model.add(Conv2D(48,(5,5),activation='relu'))
    model.add(MaxPooling2D(strides=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(strides=(2,2)))

    # Because the output size of above layer is 2x17x64, this layer must be deleted.
    #model.add(Conv2D(64,(3,3),activation='relu'))
    # model.add(Dropout(0.5))
    #model.add(MaxPooling2D(strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':
    from sklearn.utils import shuffle
    epochs = 20

    # load training data
    X_train, y_train = LoadTrainingData(['datasets/track1/0/'], True)

    #X_train, y_train = LoadTrainingData(['datasets/track2/0/','datasets/track2/1/'], False)
    
    # setup keras and create cnn model
    SetupKeras()
    model = CreateModel(X_train[0].shape)

    # train
    for epoch in range(epochs):
        # At first, shuffle training data and validation data (X_train include both training data and validation data).
        X_train, y_train = shuffle(X_train, y_train)
        model.fit(X_train, y_train, validation_split=0.2,epochs=epoch+1, initial_epoch=epoch)
    
    model.save('model.h5')

