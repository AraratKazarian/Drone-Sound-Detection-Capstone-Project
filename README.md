---
project: "Real-Time Drone Sound Detection"
authors:
  - "Eduard Petrosyan"
  - "Ararat Kazarian"
supervisor: "Gagik Khalafyan"
affiliation: "American University of Armenia, BS in Data Science"
date: "2025-05-09"
---

# Real-Time Drone Sound Detection

## 📖 Overview
Small drones are increasingly used in both military and civilian contexts, but are hard to spot visually or by radar.  
This project builds a **real-time**, **sound-based** drone detector using machine learning.

- **Input**: 1 second audio clips (16 kHz)  
- **Output**: binary label: `drone` / `non-drone`  
- **Core model**: a compact 2-layer CNN on MFCC feature “images”  

## 📂 Repository Structure
```bash
├── data/                         # Raw & segmented audio
│   ├── drone/                    # 1s drone clips
│   └── non_drone/                # 1s non-drone clips
├── feature_extraction/           
│   ├── audio_features.npz        # Precomputed MFCCs
│   └── feature_extraction.ipynb  # Notebook to generate MFCCs
├── model_development/
│   ├── model_development.ipynb   # Training & evaluation pipeline
│   └── audio_cnn_weights.pth     # Final CNN weights
├── real_time_app/
│   └── audio_input.py            # Live capture & GUI
├── results/
│   ├── figures/                  # Plots & spectrograms
│   └── tables/                   # LaTeX tables
├── README.md                     # ← you are here
├── LICENSE
└── .gitignore
```

## 📊 Data

- **DroneAudioDataset (GitHub)**
  - 1 332 drone clips (Parrot Mambo, Bebop)  
  - 10 372 non-drone clips (wind, vehicles, voices, etc.)

- **Augmentation & Extras**
  - Manual laptop‐mic recordings  
  - Freesound.org & BBC SFX grabs  
  - All clips resampled to 16 kHz, segmented into 1 s WAVs  

- **Final counts after augmentation:**
  - **Drone**: 2 400 → 10 925  
  - **Non-drone**: 13 806 → 11 767  

- **Features**
  - 13 MFCCs × 32 time-frames per clip  
  - Saved in `audio_features.npz`  

## ⚙️ Methods

### 1. Data Preparation
- Resample → Segment → MFCC  
- Stratified 80/20 train/test split  
- Drone class augmented (time-stretch, pitch-shift, noise)  

### 2. Model Selection
We compared six classifiers on the same MFCC inputs:  
- **K-Nearest Neighbors (KNN):** simple, clusters intact  
- **Logistic Regression:** linear baseline  
- **SVM:** max-margin (RBF kernel)  
- **Random Forest:** ensemble of trees, robust to noise  
- **XGBoost:** gradient-boosted trees, top accuracy  
- **CNN:** 2D conv on MFCC maps, best generalization  

Each non-CNN model was tuned via grid-search + 5-fold CV.

### 3. Evaluation & Tuning
| Model                 | Accuracy | F1-Score |
|-----------------------|----------|----------|
| KNN                   | 0.970    | 0.970    |
| Logistic Regression   | 0.889    | 0.889    |
| SVM                   | 0.970    | 0.970    |
| Random Forest         | 0.980    | 0.980    |
| XGBoost               | 0.983    | 0.983    |
| **CNN**               | 0.971    | 0.971    |

> **Why CNN in deployment?**  
> • Learns local time-frequency filters (propeller harmonics)  
> • Generalizes to new noise patterns  
> • Very fast on modern hardware (real-time feasible)  

### 4. Real-Time App
- Captures 1 s audio windows from mic  
- Extracts MFCC → CNN → predicts “drone” vs. “non-drone”  
- Displays live mel-spectrogram + probability bars  

## 🔎 Findings
- **Field test (Azatazen Polygon):**  
  Detected FPV drone reliably at 30–40 m in outdoor noise.  

## 🚀 Future Work
- Local testing in Armenia (schools, events, campuses)  
- Portable unit (microcontroller + mic)  
- Expanded dataset: windy, mountainous, urban canyons  
- Partnerships: civic bodies, heritage sites  
- Collab with national security for controlled “battlefield” drills  

## 🙏 Acknowledgments
- **Gagik Khalafyan (Supervisor):** expert guidance & feedback  
- **AUA Drone Club:** FPV drone access, test setups  
- **American University of Armenia:** resources & support  


