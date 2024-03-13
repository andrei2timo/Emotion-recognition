import pandas as pd
import cv2
import numpy as np


dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)

"""
    Function to load the FER2013 dataset.

    This function reads the CSV file containing pixel values of images,
    reshapes and resizes the images, and extracts emotion labels.

    :return: faces - array of preprocessed images, emotions - one-hot encoded emotion labels
"""
def load_fer2013():
    data = pd.read_csv(dataset_path) # Read the CSV file

    # Extract pixel values and reshape images
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []

    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    # Extract emotion labels and one-hot encode them
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    return faces, emotions

"""
    Function to preprocess input images.

    This function normalizes pixel values and optionally applies V2 preprocessing.

    :param x: input images
    :param v2: if True, apply V2 preprocessing
    :return: preprocessed input images
"""
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x