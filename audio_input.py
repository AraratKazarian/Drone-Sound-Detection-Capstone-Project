import sounddevice as sd
import numpy as np
import pickle
import python_speech_features as psf  # If you used MFCCs

# Load your trained model
with open("clf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Parameters
CHUNK_DURATION = 1  # seconds
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Preprocessing function
def extract_features(audio_data, sample_rate):
    mfccs = psf.mfcc(audio_data, samplerate=sample_rate, numcep=13)
    return np.mean(mfccs, axis=0).reshape(1, -1)

# Callback for real-time processing
def callback(indata, frames, time, status):
    if status:
        print(status)

    # Convert audio to 1D array
    audio_np = indata[:, 0]

    # Normalize if needed
    audio_np = audio_np.astype(np.float32)
    audio_np /= np.max(np.abs(audio_np), axis=0)

    # Extract features
    features = extract_features(audio_np, SAMPLE_RATE)

    # Predict
    prediction = model.predict(features)

    if prediction[0] == "drone":
        print("🚨 Drone sound detected!")

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
