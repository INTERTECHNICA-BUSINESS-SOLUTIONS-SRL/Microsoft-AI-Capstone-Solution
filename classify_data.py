#!/usr/bin/env python

"""
Classifies the test data and generates the submissions.
"""
import logging

import numpy as np
import numpy.random as rnd
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

from PIL import Image, ImageFilter

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

__author__ = 'Marin Iuga'
__copyright__ = 'Copyright 2018, Marin Iuga / Intertechnica Business Solutions SRL'
__credits__ = ['Marin Iuga']
__license__ = 'MIT'
__version__ = '1.0'
__maintainer__ = 'Marin Iuga'
__email__ = 'marin.iuga@intertechnica.com'
__status__ = 'Production'

#configuration data 
image_height = 128//1
image_width = 2*118//1
training_image_count = 988
testing_image_count = 659
classes_count = 11

data_root_path = './data/'

training_image_data_file_path = data_root_path + 'image_train.data'
training_labels_data_file_path = data_root_path + 'image_train_labels.csv'
testing_data_file_path = data_root_path + 'image_test.data'

testing_submission_file_path = data_root_path + 'submission_format.csv'
submission_results_file_path =  data_root_path + 'submission_results.csv'

def read_image(p_image_data_file_path, p_position, p_image_width, p_image_height) :
    """
    Reads an image from an image data from a image data repository @see prepare_data.py

    @params:
        p_image_data_file_path - Required : the image data file path (String)
        p_position - Required : second image source (int)
        p_image_width - Required: the image width (int)
        p_image_height - Required: the image height (int)
    
    @returns:
        The image data (array)    
    """  
    with open(p_image_data_file_path, "rb") as image_file :
        image_file.seek(p_position * p_image_height* p_image_width)
        data = image_file.read(p_image_height * p_image_width)
    
        data_b = np.frombuffer(data, dtype=np.uint8)

    return np.asarray(data_b)

def process_images(p_images, p_image_width, p_image_height) :
    """
    Processes a set of images so it can be classified by the neurals network model

    @params:
        p_images - Required : the images to process (String)
        p_image_width - Required: the image width (int)
        p_image_height - Required: the image height (int)
    """  
    #reshape according to inputs accepted by a Conv2d layer
    processed_images = p_images.reshape(p_images.shape[0], p_image_height, p_image_width, 1)

    #data normalization to max value (0-255 grayscale values)
    processed_images = (processed_images * 1.0) /255
 
    return processed_images
  
def read_labels(p_labels_file_path) :
    """
    Reads the extracted training labels @see prepare_data.py

    @params:
        p_labels_file_path - Required : the data file path (String)
    @returns:
        A dataframe containing the read labels with the column [id] for ordinal id and [label] for the label value    
    """ 
    labels = pd.read_csv(p_labels_file_path, header= None)
    labels.columns = ["id", "label"]
  
    return labels

def process_labels(p_labels) :
    """
    Processes the read labels

    @params:
        p_labels - Required: the read labels (array)
    @returns:
        The processed labels (binarization - one hot-encoded)    
    """
    processed_labels = LabelBinarizer().fit_transform(p_labels)
    
    return processed_labels

def generate_train_set(
    p_image_training_data_file_path, 
    p_labels_file_path, 
    p_train_set_size, 
    p_image_width, 
    p_image_height
) :
    """
    Generates the training data set

    @params:
        p_image_training_data_file_path - Required: the training image data file path (String)
        p_labels_file_path - Required: the labels file path (String)
        p_train_set_size - Required: the size of the training set (int)
        p_image_width - Required: the image width (int)
        p_image_height - Required: the image height (int)

    @returns:
        (train_labels_processed, train_images_processed) tuple wiht the the processed train labels (array) 
        and the processed train images (array)
    """
    labels = read_labels(p_labels_file_path)
    
    labels_batch = np.zeros(p_train_set_size)
    labels_batch = labels["label"][0:p_train_set_size].values

    images_batch = []
  
    for i in range(0, p_train_set_size) :
        image_data = read_image(p_image_training_data_file_path, i, p_image_width, p_image_height)
        images_batch.append(image_data.reshape(p_image_height, p_image_width))
  
    train_labels_processed = process_labels(labels_batch)
  
    train_images_processed = process_images(np.array(images_batch), p_image_width, p_image_height)
  
    return train_labels_processed, train_images_processed

def generate_test_set(
    p_test_image_data_file_path, 
    p_test_set_size, 
    p_image_width, 
    p_image_height
) :
    """
    Generates the test data set

    @params:
        p_test_image_data_file_path - Required: the testing image data file path (String)
        p_test_set_size - Required: the size of the testing set (int)
        p_image_width - Required: the image width (int)
        p_image_height - Required: the image height (int)

    @returns:
        test_images_processed the processed test images (array)
    """
    images_batch = []

    for i in range(0, p_test_set_size) :
        image_data = read_image(p_test_image_data_file_path, i, p_image_width, p_image_height)
        images_batch.append(image_data.reshape(p_image_height, p_image_width))

    test_images_processed = process_images(np.array(images_batch), p_image_width, p_image_height)

    return test_images_processed  
  
  
def create_model(p_image_width, p_image_height, p_num_classes) :
    """
    Creates the compiled model for image classification.

    @params:
        p_image_width - Required: the image width (int)
        p_image_height - Required: the image height (int)
        p_num_classes - Required: the number of classes

    @returns:
      The created and compiled model (Model)        
    """
    input_shape = (p_image_height, p_image_width, 1)

    #we will use a sequential model for training 
    model = Sequential()
	
    #CONV 3x3x32 => RELU => NORMALIZATION => MAX POOL 3x3 block
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    #CONV 3x3x64 => RELU => NORMALIZATION => MAX POOL 2x2 block
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #CONV 3x3x128 => RELU => NORMALIZATION => MAX POOL 2x2 block
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #FLATTEN => DENSE 1024 => RELU => NORMALIZATION block
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    #final DENSE => SOFTMAX block for multi-label classification
    model.add(Dense(p_num_classes))
    model.add(Activation("softmax"))

    #using categorical_crossentropy loss function with adam optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def train_model(
    p_model, 
    p_training_image_data, 
    p_trainging_labels, 
    p_batch_size = 32, 
    p_epochs_to_train = 50, 
    p_verbose_level = 2
) :
    """
    Trains the model using the train image data and train labels.
    
    @parameters:
      p_model - Required: the Keras model to be trained (Model)
      p_training_image_data - Required: the image data used for training (array)
      p_training_labels - Required: the training labels used fo training (array)
      p_batch_size - Optional, default 32: the batch size used for training (int)
      p_epochs_to_train - Optional, default 50: number of training epochs (int)
      p_verbose_level - Optional, default 2: the Keras verbose level (int)
    
    @returns:
      The trained model (Model)
    """    
    p_model.fit(
        x = p_training_image_data, 
        y = p_trainging_labels, 
        batch_size = p_batch_size, 
        epochs = p_epochs_to_train,
        shuffle = True,
        verbose = p_verbose_level    
    )
    
    return p_model

def predict_labels(p_model, p_test_image_data, p_batch_size = 32) :
    """
    Predicts the labels associated with the test data.
    
    @parameters:
      p_model - Required: the Keras model to be used (Model)
      p_test_image_data - Required: the image data used for testing (array)
      p_batch_size - Optional, default 32: the batch size used for training (int)
    
    @returns:
      The predicted label (array)
    """      
    labels = p_model.predict_classes(p_test_image_data, p_batch_size)
  
    return labels

def write_results(
    p_testing_submission_file_path, 
    p_submission_results_file_path, 
    p_results
) :
    """
    Writes the result to the output file.
    
    @parameters:
      p_testing_submission_file_path - Required: the path to the submission format (String)
      p_submission_results_file_path - Required: the path to the output file (String)
      p_results - Required: the results to be written in the outut file (array)
    """     
    submission_structure = pd.read_csv(p_testing_submission_file_path)
    submission_structure['appliance'] = p_results
    submission_structure.to_csv(p_submission_results_file_path, index=False)
  
def main():
    logging.basicConfig(level=logging.INFO)
    
    #prepare training data
    logging.info('Reading training data ...')
    train_labels, train_images = generate_train_set(
        training_image_data_file_path, 
        training_labels_data_file_path, 
        training_image_count, 
        image_width, 
        image_height
    )
    logging.info('Reading training data DONE')
    
    #create and train model
    logging.info('Creating model ...')
    model = create_model (image_width, image_height, classes_count)
    logging.info('Creating model DONE')

    logging.info('Training model ... ')
    model = train_model(model, train_images, train_labels, p_epochs_to_train = 50)
    logging.info('Training model DONE')
    
    #create test data
    logging.info('Reading testing data ...')
    test_images = generate_test_set(
      testing_data_file_path, 
      testing_image_count, 
      image_width, 
      image_height
    )
    logging.info('Reading testing data DONE')
    
    #predict labels for test data
    logging.info('Predicting test data classes ...')
    result = predict_labels(model, test_images)
    logging.info('Predicting test data classes DONE')
    
    #write results
    logging.info('Writing results ...')
    write_results(
        testing_submission_file_path, 
        submission_results_file_path, 
        result
    )
    logging.info('Writing results DONE')
    
    
if __name__ == '__main__':
    main()