# Real-Time Drone Sound Detection

This repository contains code, data, and results for a **real-time drone sound detection** system. We extract audio features from 1 s WAV clips, train multiple classifiers (including a CNN), and deploy a live inference script to flag drone activity from microphone input.

---

## 🚀 Quickstart

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
Prepare raw data

Place 1 s WAV files under:

kotlin
Copy
Edit
data/
  ├─ drone/
  └─ non_drone/
Extract features

bash
Copy
Edit
cd feature_extraction
jupyter notebook feature_extraction.ipynb
– Generates audio_features.npz.

Train & evaluate

bash
Copy
Edit
cd ../model_development
jupyter notebook model_development.ipynb
– Produces performance tables, plots, and audio_cnn_weights.pth.

Run live detection

bash
Copy
Edit
python audio_input.py
– Streams mic input, extracts MFCCs, and prints “DRONE” / “NO DRONE”.

📂 Repository Structure

├── data/
│   ├── drone/                   # Raw drone audio clips (.wav)
│   └── non_drone/               # Raw non-drone audio clips (.wav)
│
├── feature_extraction/
│   ├── feature_extraction.ipynb # Extracts 1 s segments → MFCCs → audio_features.npz
│   └── audio_features.npz       # Compressed MFCC arrays (train & test splits)
│
├── model_development/
│   ├── model_development.ipynb  # Trains KNN, Logistic, SVM, RF, XGBoost, and CNN
│   └── audio_cnn_weights.pth    # Saved PyTorch CNN weights
│
├── results/
│   ├── images/                  # UI screenshot, sample spectrograms
│   ├── plots/                   # Performance & feature visualizations
│   └── tables/                  # CSV & LaTeX tables of counts & metrics
│
├── audio_input.py               # Live mic capture → feature → CNN inference
├── requirements.txt             # `pip install` dependencies
└── README.md                    # This file
🔍 Data & Features
Raw audio: 16 kHz-resampled, 1 s segments, labels drone / non_drone.

Feature extraction:

13 MFCCs per frame

Stored in audio_features.npz with train/test splits.

🤖 Models Evaluated
Model	Accuracy	F1-Score
K-Nearest Neighbors	0.970	0.970
Logistic Regression	0.889	0.889
Support Vector Machine	0.970	0.970
Random Forest	0.980	0.980
XGBoost	0.983	0.983
CNN (PyTorch)	0.971	0.971

Final choice: CNN for its ability to learn spatio-temporal audio patterns and run efficiently in real time.

🎛 Live Detection
The audio_input.py script:

Captures continuous 1 s audio windows from your default microphone.

Computes MFCCs on the fly.

Loads audio_cnn_weights.pth.

Outputs “DRONE” or “NO DRONE” every second.

📄 License
This project is licensed under the MIT License. See LICENSE for details.
