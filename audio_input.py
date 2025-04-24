import sounddevice as sd
import numpy as np
import pickle
import python_speech_features as psf  # If you used MFCCs
import librosa
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Load your trained model
with open("clf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Parameters
CHUNK_DURATION = 1  # seconds
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Normalize audio function
def normalize_audio(audio_np):
    audio_np = audio_np.astype(np.float32)
    if np.max(np.abs(audio_np)) > 0:
        audio_np /= np.max(np.abs(audio_np))
    return audio_np

# Preprocessing function
# Real-time feature extraction with librosa (to match training)
def extract_features_librosa(audio_data, sample_rate):
    # Use librosa to extract MFCC from raw audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs, axis=1).reshape(1, -1)

# Callback for real-time processing
def callback(indata, frames, time, status):
    if status:
        print(status)

    # Convert audio to 1D array
    audio_np = indata[:, 0]

    # Normalize the audio (if needed, based on how you trained)
    audio_np = normalize_audio(audio_np)

    # Extract features using Librosa (to match training)
    features = extract_features_librosa(audio_np, SAMPLE_RATE)

    # Predict
    prediction = model.predict(features)
    print(prediction)
    if prediction == 0:
        print("🚨 Drone sound detected!")
    else:
        print("✅ Non-drone sound detected.")

print("Listening... Press Ctrl+C to stop.")

try:
    with sd.InputStream(callback=callback,
                        channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        blocksize=CHUNK_SIZE,
                        dtype='int16'):
        while True:
            sd.sleep(1000)  # Keeps the loop alive

except KeyboardInterrupt:
    print("Stopped.")