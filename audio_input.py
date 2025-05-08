import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pyaudio
import tkinter as tk
from tkinter import ttk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Constants
MAX_LEN = 32
N_MFCC = 13
SAMPLE_RATE = 16000
CHUNK_DURATION = 1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
CHANNELS = 1
FORMAT = pyaudio.paInt16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
class AudioCNN(nn.Module):
    def __init__(self, max_len=32, n_mfcc=13):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d((2, 2))
        conv_out_h = (max_len - 4) // 2
        conv_out_w = (n_mfcc - 4) // 2
        self.fc1 = nn.Linear(16 * conv_out_h * conv_out_w, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load model
model = AudioCNN(MAX_LEN, N_MFCC).to(DEVICE)
model.load_state_dict(torch.load("model_development/audio_cnn_weights.pth", map_location=DEVICE))
model.eval()

# Audio & MFCC extraction
p = pyaudio.PyAudio()


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfcc.shape[0] < MAX_LEN:
        pad = np.zeros((MAX_LEN - mfcc.shape[0], N_MFCC))
        mfcc = np.vstack([mfcc, pad])
    else:
        mfcc = mfcc[:MAX_LEN]
    return mfcc


def process_audio(data, bar_drone, bar_non_drone, ax, canvas):
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    mfcc = extract_mfcc(audio)
    tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    bar_drone["value"] = probs[1] * 100
    bar_non_drone["value"] = probs[0] * 100

    # Plot Mel spectrogram
    ax.clear()
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    ax.imshow(mel_db, origin='lower', aspect='auto', cmap='viridis')
    ax.set_title("Mel-Spectrogram")
    ax.axis('off')
    canvas.draw()


def audio_loop(bar_drone, bar_non_drone, ax, canvas):
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        process_audio(data, bar_drone, bar_non_drone, ax, canvas)

# GUI setup
BAR_LENGTH = 500
NUM_TICKS = 10
TOP_OFFSET = 30

root = tk.Tk()
root.title("Drone Audio Classifier")
root.configure(bg='white')
root.minsize(800, 600)

# Title frame
title_frame = tk.Frame(root, bg='white')
title_frame.grid(row=0, column=0, sticky='ew')
title_frame.columnconfigure(0, weight=1)
tk.Label(title_frame,
         text="Real-Time Drone Sound Detection",
         font=('Helvetica', 34, 'bold'),
         bg='white', fg='black').grid(row=0, column=0, pady=20)

# Main content frame
main_frame = tk.Frame(root, bg='white')
main_frame.grid(row=1, column=0, sticky='nsew')
root.rowconfigure(1, weight=1)
root.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)

# Centering frame
center_frame = tk.Frame(main_frame, bg='white')
center_frame.grid(row=0, column=0, sticky='nsew')
center_frame.rowconfigure(0, weight=1)
center_frame.columnconfigure(0, weight=1)

# 3-column layout
content_frame = tk.Frame(center_frame, bg='white')
content_frame.grid(row=0, column=0, padx=40, pady=40, sticky='nsew')
content_frame.rowconfigure(0, weight=1)
content_frame.columnconfigure(0, weight=1)
content_frame.columnconfigure(1, weight=0)  # fix spectrogram column
content_frame.columnconfigure(2, weight=1)

# Play button
play_button_frame = tk.Frame(content_frame, bg='white', width=180, height=180)
play_button_frame.grid(row=0, column=0, padx=20, pady=20)
play_button_frame.grid_propagate(False)
tk.Button(play_button_frame, text='▶', font=('Helvetica', 60),
          bg='gray', fg='white', bd=0, width=2, height=1).pack(expand=True, fill='both')

# Fixed‐size spectrogram plot
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fixed_width, fixed_height = 960, 640
canvas_frame = tk.Frame(content_frame, bg='white',
                        width=fixed_width, height=fixed_height)
canvas_frame.grid(row=0, column=1, padx=20, pady=20)
canvas_frame.grid_propagate(False)
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill='both', expand=True)

# Progress‐bar styles
style = ttk.Style()
style.theme_use('default')
style.configure("red.Vertical.TProgressbar",
                troughcolor='white', background='red',
                bordercolor='black', thickness=40)
style.configure("green.Vertical.TProgressbar",
                troughcolor='white', background='green',
                bordercolor='black', thickness=40)

# Bar container with centering rows
bar_frame = tk.Frame(content_frame, bg='white')
bar_frame.grid(row=0, column=2, padx=20, pady=20, sticky='nsew')
bar_frame.columnconfigure(0, weight=1)
bar_frame.columnconfigure(1, weight=1)
bar_frame.rowconfigure(0, weight=1)
bar_frame.rowconfigure(1, weight=0)
bar_frame.rowconfigure(2, weight=1)

# Function to draw a bar group
def create_bar_group(parent, text, style):
    group = tk.Frame(parent, bg='white')
    group.grid(row=1, column=text=='Non-Drone', sticky='nsew', padx=10, pady=10)
    # Label + bar
    bar_frame = tk.Frame(group, bg='white')
    bar_frame.grid(row=0, column=0, sticky='n')
    tk.Label(bar_frame, text=text,
             fg=('red' if text=='Drone' else 'green'),
             bg='white', font=('Helvetica', 14, 'bold')).pack()
    bar = ttk.Progressbar(bar_frame, orient='vertical',
                          length=BAR_LENGTH, mode='determinate',
                          style=style, maximum=100)
    bar.pack(fill='y', expand=True)
    # Ticks
    tick_canvas = tk.Canvas(group, width=60, height=BAR_LENGTH+TOP_OFFSET*2,
                             bg='white', highlightthickness=0)
    tick_canvas.grid(row=0, column=1, sticky='n', padx=(5, 0))
    for idx, perc in enumerate(range(100, -1, -10)):
        y = idx * (BAR_LENGTH // NUM_TICKS) + TOP_OFFSET
        tick_canvas.create_line(0, y, 20, y, fill='black', width=2)
        tick_canvas.create_text(25, y, anchor='w',
                                text=f"{perc}%", font=('Helvetica', 9))
    return bar

# Create both bars
bar_drone = create_bar_group(bar_frame, 'Drone', 'red.Vertical.TProgressbar')
bar_non_drone = create_bar_group(bar_frame, 'Non-Drone', 'green.Vertical.TProgressbar')

bar_drone['value'] = 50
bar_non_drone['value'] = 50

# Audio loop starter
def start_audio():
    Thread(target=audio_loop, args=(bar_drone, bar_non_drone, ax, canvas)).start()

play_button_frame.children['!button'].config(command=start_audio)

root.mainloop()