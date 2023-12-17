# Speech-Emotion-Recognition
This GitHub repository contains code for building and evaluating a Convolutional Neural Network (CNN) model to recognize emotions from audio data. The model is trained and tested on a dataset consisting of audio recordings of various emotions, including anger, disgust, fear, happiness, neutral, sadness, and surprise.

## Overview

This GitHub repository contains code for building and evaluating a Convolutional Neural Network (CNN) model to recognize emotions from audio data. The model is trained and tested on a dataset consisting of audio recordings of various emotions, including anger, disgust, fear, happiness, neutral, sadness, and surprise.

**Dataset**
The dataset used for this project is a combination of three different emotion databases: RAVDESS, TESS, and SAVEE. Each dataset is processed and combined into a single DataFrame containing file paths, audio durations, dataset labels, and emotion labels.

## RAVDESS Dataset

Audio files from the RAVDESS dataset are processed.
Emotions are labeled as 'neutral,' 'calm,' 'happy,' 'sad,' 'angry,' 'fear,' 'disgust,' or 'surprise.'
TESS Dataset
Audio files from the TESS dataset are processed.
Emotions are labeled as 'angry,' 'disgust,' 'fear,' 'happy,' 'neutral,' 'sad,' or 'surprise.'
'ps' (pleasant surprise) is replaced with 'surprise' in the emotion labels.
SAVEE Dataset
Audio files from the SAVEE dataset are processed.
Emotions are labeled as 'angry,' 'disgust,' 'fear,' 'happy,' 'neutral,' 'sad,' or 'surprise.'
Preprocessing
Feature Extraction
Mel-frequency cepstral coefficients (MFCCs) are extracted from the audio files to serve as features for the CNN model.
The MFCCs are resized to ensure uniform dimensions for model input.
Data Splitting
The dataset is split into training, validation, and test sets.
The training set consists of 70% of the data, the validation set consists of 30%, and the test set consists of 10%.

## Model Architecture

The CNN model consists of several layers:

Convolutional Layer 1: 64 filters, 5x5 kernel size, ReLU activation, and max pooling.
Batch Normalization.
Convolutional Layer 2: 32 filters, 4x4 kernel size, ReLU activation, and max pooling.
Batch Normalization.
Flatten Layer.
Dropout Layer (to prevent overfitting).
Fully Connected Dense Layer 1: 128 nodes and ReLU activation.
Dropout Layer.
Fully Connected Dense Layer 2: 64 nodes and ReLU activation.
Dropout Layer.
Output Layer: 7 nodes with softmax activation for predicting 7 emotion classes.
Evaluation
The model's performance is evaluated using the test set, and the predictions are compared to the ground truth labels. Two confusion matrices are generated: one displaying counts and another displaying ratios. These matrices help visualize the model's accuracy in recognizing different emotions.

## Usage

To use this code, follow these steps:

Prepare your audio dataset and update the file paths accordingly.
Install the required libraries mentioned at the beginning of the code.
Run the code sections sequentially in a Python environment (e.g., Jupyter Notebook or Google Colab).
Feel free to customize the model architecture, hyperparameters, or preprocessing steps to adapt it to your specific audio emotion recognition task.


##Acknowledgments

Special thanks to the creators of the RAVDESS, TESS, and SAVEE datasets for providing valuable resources for emotion recognition research.






