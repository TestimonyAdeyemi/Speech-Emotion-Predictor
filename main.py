import streamlit as st
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Zeros
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('cnn.h5')  # Update with the correct path

# Function to predict emotion from audio
# Function to preprocess and predict emotion from audio


import librosa
import numpy as np

import librosa
import numpy as np
import os


# Emotions in the RAVDESS dataset, different numbers represent different emotion
emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}



def extract_feature(data, sr, mfcc, chroma, mel):

    """
    extract features from audio files into numpy array

    Parameters
    ----------
    data : np.ndarray, audio time series
    sr : number > 0, sampling rate
    mfcc : boolean, Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
    chroma : boolean, pertains to the 12 different pitch classes
    mel : boolean, Mel Spectrogram Frequency

    """

    if chroma:
        stft = np.abs(librosa.stft(data))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        #mel = np.mean(librosa.feature.melspectrogram(data, sr=sr).T,axis=0)
        result = np.hstack((result, mel))

    return result



def noise(data, noise_factor):

    """
    add random white noises to the audio

    Parameters
    ----------
    data : np.ndarray, audio time series
    noise_factor : float, the measure of noise to be added

    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise

    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift(data, sampling_rate, shift_max, shift_direction):

    """
    shift the spectogram in a direction

    Parameters
    ----------
    data : np.ndarray, audio time series
    sampling_rate : number > 0, sampling rate
    shift_max : float, maximum shift rate
    shift_direction : string, right/both

    """
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0

    return augmented_data





def load_single_data(file):
    x, y = [], []
    # file_name = os.path.basename(file)
    # emotion = emotions[file_name.split("-")[2]]
    data, sr = librosa.load(file)
    feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    # y.append(emotion)
    return np.array(x), y





emotion_images = {
    'neutral': 'image\neutral.png',
    'calm': 'image\calm.png',
    'happy': 'image\happy.png',
    'sad': 'image\sad.png',
    'angry': 'image\angry.png',
    'fearful': 'image\fearful.png',
    'disgust': 'image\disgust.png',
    'surprised': 'image\surprised.png',
}


model = load_model('cnn.h5')  # Update with the correct path

# Function to preprocess and predict emotion from audio
def predict_emotion(file):
    XX, yy = load_single_data(file)
    XXTemp = np.expand_dims(XX, axis=2)
    prediction = model.predict(XXTemp)
    print(prediction)

    # Assuming prediction is an array of probabilities
    predicted_class = np.argmax(prediction)
    #predicted_emotion = emotions.get(str(predicted_class + 1), 'Unknown')

    prediction_array = np.array(prediction)

    # Emotion labels
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    # Get the index with the highest probability
    predicted_emotion_index = np.argmax(prediction_array)

    # Get the corresponding emotion label
    predicted_emotion = emotion_labels[predicted_emotion_index]

    return predicted_emotion

# Streamlit app
st.title("Speech Emotion Recognition")

# File upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Predict emotion and display the result
    predicted_emotion = predict_emotion(uploaded_file)
    # Example prediction array

    # Print the predicted emotion
    st.success(f"Predicted emotion: {predicted_emotion}")

    if predicted_emotion in emotion_images:
        image_url = emotion_images[predicted_emotion]
        st.image(image_url, caption=f"Image for {predicted_emotion}", use_column_width=True)

