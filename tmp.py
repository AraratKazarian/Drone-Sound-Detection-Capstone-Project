import pyaudio
import wave
import os
import time

# Recording settings
FORMAT = pyaudio.paInt32  # 16-bit is common and smaller in size
CHANNELS = 1
RATE = 16000  # 16 kHz
CHUNK = 1024
RECORD_SECONDS = 1
NUM_RECORDINGS = 100
OUTPUT_DIR = "silence_recordings"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start loop to record NUM_RECORDINGS files
for i in range(NUM_RECORDINGS):
    print(f"🎙️ Recording {i + 1}/{NUM_RECORDINGS}...")

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    filename = f"silence_{i}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    wf = wave.open(filepath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    time.sleep(0.2)  # short pause to reset between recordings

audio.terminate()
print(f"✅ Done. Saved {NUM_RECORDINGS} silence clips in '{OUTPUT_DIR}'")
