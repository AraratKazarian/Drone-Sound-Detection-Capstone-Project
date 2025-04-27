import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import librosa
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ————————————
# 1) Define your CNN exactly as before
# ————————————
class AudioCNN(nn.Module):
    def __init__(self, max_len=32, n_mfcc=40):
        super().__init__()
        # note: max_len and n_mfcc must match what you used during training
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d((2,2))
        # compute flattened size after conv+pool:
        conv_out_h = (max_len - 4) // 2
        conv_out_w = (n_mfcc - 4) // 2
        self.fc1 = nn.Linear(16 * conv_out_h * conv_out_w, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # x: (batch, 1, max_len, n_mfcc)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ————————————
# 2) Load your trained weights
# ————————————
DEVICE = torch.device('cpu')     # real-time on CPU
MAX_LEN = 32                     # frames (as in training)
N_MFCC  = 40                     # mfcc bands

model = AudioCNN(max_len=MAX_LEN, n_mfcc=N_MFCC).to(DEVICE)
state = torch.load("audio_cnn_weights.pth", map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ————————————
# 3) Real-time params & helpers
# ————————————
SAMPLE_RATE    = 16000
CHANNELS       = 1
CHUNK_DURATION = 1          # 1 second
CHUNK_SIZE     = SAMPLE_RATE * CHUNK_DURATION

def normalize_audio(a):
    a = a.astype(np.float32)
    m = np.max(np.abs(a))
    return a/m if m>0 else a

def extract_mfcc_sequence(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_len=MAX_LEN):
    # get MFCC frames & transpose to (time_steps, n_mfcc)
    mf = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    # pad or truncate to max_len
    if mf.shape[0] < max_len:
        pad = np.zeros((max_len - mf.shape[0], n_mfcc), dtype=np.float32)
        mf = np.vstack([mf, pad])
    else:
        mf = mf[:max_len]
    return mf

# ————————————
# 4) Callback & stream
# ————————————
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio = normalize_audio(indata[:,0])
    mf_seq = extract_mfcc_sequence(audio)
    # shape (1, 1, max_len, n_mfcc)
    tensor = torch.from_numpy(mf_seq).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()

    if pred == 1:
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
            sd.sleep(1000)
except KeyboardInterrupt:
    print("Stopped.")
