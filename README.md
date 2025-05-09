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
