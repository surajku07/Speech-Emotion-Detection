import streamlit as st
import numpy as nppip
import librosa
from io import BytesIO
from keras.models import load_model
import tempfile
import os

# Load the pre-trained model
model = load_model("speech.h5")

# Function to extract MFCC features from audio input
def extract_mfcc_from_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Streamlit App
st.title("Emotion Classifier for Audio")

# File Upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())

    st.audio(temp_file.name, format="audio/wav")

    # Extract MFCC features
    input_mfcc = extract_mfcc_from_audio(temp_file.name)

    # Reshape input for model prediction
    input_mfcc = np.expand_dims(input_mfcc, axis=0)
    input_mfcc = np.expand_dims(input_mfcc, axis=-1)

    # Make prediction
    prediction = model.predict(input_mfcc)

    # Display predicted emotion
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    predicted_emotion = emotions[np.argmax(prediction)]
    st.write(f"Predicted Emotion: {predicted_emotion}")

    # Remove temporary file
    os.remove(temp_file.name)