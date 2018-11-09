#!/usr/bin/env python

"""
Prepares data for easier processing by the neural net classifier.
"""

import math
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

__author__ = 'Marin Iuga'
__copyright__ = 'Copyright 2018, Marin Iuga / Intertechnica Business Solutions SRL'
__credits__ = ['Marin Iuga']
__license__ = 'MIT'
__version__ = '1.0'
__maintainer__ = 'Marin Iuga'
__email__ = 'marin.iuga@intertechnica.com'
__status__ = 'Production'

#Configuration/parameters
classes_count = 11
image_width = (2*118)//1
image_height = 128//1
image_data_size = image_width*image_height
channels = 1

input_data_file_path = './data/data-release.zip'

training_data_file_path = './data/image_train.data'
labels_data_file_path = './data/image_train_labels.csv'
testing_data_file_path = './data/image_test.data'
testing_submission_file_path = './data/submission_format.csv'

def compose_train_image(p_img1, p_img2) :
    """
    Creates a horizontally stacked image using two input images

    @params:
        p_img1 - Required : first image source (Image)
        p_img2 - Required : second image source (Image) 

    @returns:
        The stacked image (Image)    
    """ 

    #Stacks images horizontally (i.e. one afer another on width axis)
    img_merge_data = np.hstack([np.asarray(p_img1), np.asarray(p_img2)])
    img_merge = Image.fromarray( img_merge_data )
        
    return img_merge

def get_image_data(p_image) :
    """
    Returns a flatten array of image pixel values (1 channel gray pallete)

    @params:
        p_image - Required : the input image (Image)

    @returns:
        Flattened array of image data (array)
    """  

    #Generates image data from the received image object
    width, height = p_image.size
    data = np.asarray(p_image).reshape(height*width)
    
    return data


def create_trainining_images_data_file(p_input_data_file_path, p_training_data_file_path):
    """
    Creates training information data

    @params:
        p_input_data_file_path - Required: the input data file path (String)
        p_training_data_file_path - Required: the output training data file path (String)

    @returns:
        The extracted training labels (array)
    """  

    training_labels_file_path = 'train_labels.csv'
    
    labels = None

    with open(p_training_data_file_path, 'w+b') as data_file :
        with ZipFile(p_input_data_file_path) as data_zip:
            with data_zip.open(training_labels_file_path) as train_labels_file:
                content = train_labels_file.read()
                with BytesIO(content) as io_content:
                    train_labels = pd.read_csv(io_content)

                    max_count = train_labels.shape[0]    
                    labels = np.zeros(max_count)

                    count = 0

                    for _, row in train_labels.iterrows() :

                        with data_zip.open('train/' + str(row["id"]) + "_c.png") as c_file :
                            with BytesIO(c_file.read()) as input_buffer:
                                c_image = Image.open(input_buffer).convert("L")

                        with data_zip.open('train/' + str(row["id"]) + "_v.png") as v_file :
                            with BytesIO(v_file.read()) as input_buffer:
                                v_image = Image.open(input_buffer).convert("L")

                        image_data = get_image_data(compose_train_image(c_image, v_image))

                        labels[count] = row["appliance"]
                        data_file.write(image_data)

                        count = count + 1       

    return labels[:count]

def create_training_labels(p_labels, p_labels_data_file_path) :
    """
    Writes the training labels to a destination file

    @params:
        p_labels - Required: the array of labels (array)

    """ 

    classes = pd.DataFrame(p_labels.astype(int))
    classes.to_csv(p_labels_data_file_path, header=None)

    return


def create_testing_images_data_file(p_input_data_file_path, p_testing_data_file_path):
    """
    Creates testing information data

    @params:
        p_input_data_file_path - Required: the input data file path (String)
        p_testing_data_file_path - Required: the output testing data file path (String)

    @returns:
        The count of test images (int)
    """  

    submission_format_file_path = 'submission_format.csv'

    with open(p_testing_data_file_path, 'w+b') as data_file :
        with ZipFile(p_input_data_file_path) as data_zip:
            with data_zip.open(submission_format_file_path) as submission_format_file:
                content = submission_format_file.read()
                with BytesIO(content) as io_content:
                    submission_indexes = pd.read_csv(io_content)

                    count = 0

                    for _, row in submission_indexes.iterrows() :

                        with data_zip.open('test/' + str(row["id"]) + "_c.png") as c_file :
                            with BytesIO(c_file.read()) as input_buffer:
                                c_image = Image.open(input_buffer).convert("L")

                        with data_zip.open('test/' + str(row["id"]) + "_v.png") as v_file :
                            with BytesIO(v_file.read()) as input_buffer:
                                v_image = Image.open(input_buffer).convert("L")

                        image_data = get_image_data(compose_train_image(c_image, v_image))
                        data_file.write(image_data)

                        count = count + 1       

    return count

def create_testing_submission(p_input_data_file_path, p_testing_submission_file_path) :
    """
    Writes the submission data to a destination file

    @params:
        p_input_data_file_path - Required: the input data file path (String)
        p_testing_submission_file_path - Required: the testing submission file path (String)

    """ 
    submission_format_file_path = 'submission_format.csv'

    with ZipFile(p_input_data_file_path) as data_zip:
        with data_zip.open(submission_format_file_path) as submission_format_file:
            content = submission_format_file.read()
            with BytesIO(content) as io_content:
                submission_indexes = pd.read_csv(io_content)
                submission_indexes.to_csv(p_testing_submission_file_path, index=False)

    return

def main() :
    """
    Entry point
    """
    training_labels  = create_trainining_images_data_file(input_data_file_path, training_data_file_path)
    create_training_labels(training_labels, labels_data_file_path)
    print("Processed training images count: %d" % training_labels.shape[0])

    testing_count = create_testing_images_data_file(input_data_file_path, testing_data_file_path)
    create_testing_submission(input_data_file_path, testing_submission_file_path)

    print("Processed testing images count: %d" % testing_count)

if __name__ == '__main__':
    main()