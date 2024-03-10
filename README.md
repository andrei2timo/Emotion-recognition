# Table of Content :
1.[Description](#p1)
2.[Installations](#p2)
3.[Usage](#p3)
4.[Dataset](#p4)




<a id="p1"></a> 
# Description:

Emotion recognition is a technique used in software that allows a program to analyze the emotions on a human face using advanced image processing. This project aims to demonstrate the probabilities of various emotions present on human faces by utilizing machine learning and computer vision techniques.

## What does Emotion Recognition mean?

Emotion recognition involves analyzing facial expressions to determine the emotions being conveyed by a person. By employing sophisticated algorithms and image processing techniques, software can interpret the emotions depicted on a person's face and provide insights into their emotional state.

<a id="p2"></a>

## Installations

Install dependencies using requirements.txt
To install the necessary dependencies, run the following command:

```shell
pip install -r requirements.txt

<a id="p3"></a> 
# Usage:

The program will create a window displaying the scene captured by the webcam.

> Demo

python GUI.py

The program uses a pretrained model included in the specified path in the code file. You can choose a different model by running the train_emotion_classifier.py file and training your own model.

> If you want to train your own model, run the following command:
- Train

- python train_emotion_classifier.py


<a id="p4"></a> 
# Dataset:

I have used [this](https://www.kaggle.com/c/3364/download-all) dataset

This project uses the FER2013 dataset for emotion classification. Download the dataset and place the CSV file in the fer2013/fer2013/ directory.

-fer2013 emotion classification test accuracy: 66%


# Ongoing 
Draw emotions faces next to the detected face.
