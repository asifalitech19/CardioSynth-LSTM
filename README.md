# CardioSynth-LSTM 🫀🤖

**Predictive Arrhythmia Forecasting and Synthetic ECG Synthesis using LSTM-Autoencoders and Time-Series GANs (TimeGAN)**

This repository contains a complete biomedical AI framework designed to process physiological time-series data (Electrocardiograms). It addresses data scarcity, class imbalance, and temporal noise in long-term patient monitoring by utilizing deep sequential models. 

This project was developed as a comprehensive open-ended laboratory implementation for the Artificial Intelligence program at Iqra University.

## 📌 Project Overview
The architecture is divided into two primary pipelines and a final evaluation phase:
1. **Anomaly Detection (Path A):** An LSTM-Autoencoder that compresses normal ECG sequences and flags anomalous heartbeats (e.g., Ventricular Premature Contractions) based on high temporal reconstruction loss.
2. **Data Synthesis (Path B):** A Recurrent Time-Series Generative Adversarial Network (TimeGAN) that synthesizes realistic minority-class ECG beats to mitigate training dataset imbalance.
3. **Downstream Classification:** A secondary LSTM classifier trained on augmented (real + synthetic) data, benchmarked against a baseline model trained strictly on imbalanced real data.

## 📂 Repository Structure
The codebase is structured into modular directories as per the project requirements:

```text
CardioSynth-LSTM/
├── data_loader/
│   └── physionet_ingestion.py      # MIT-BIH downloading, bandpass filtering (0.5-45 Hz), and Z-score normalization
├── models/
│   ├── LSTM_Autoencoder.py         # Sequence-to-sequence autoencoder with recurrent dropout
│   └── TimeGAN.py                  # Recurrent Generator and Discriminator with Wasserstein loss
├── evaluation/
│   ├── classification_eval.py      # Downstream classifier and accuracy comparison
│   └── visual_metrics.py           # Code to generate loss curves and ECG waveform plots
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
