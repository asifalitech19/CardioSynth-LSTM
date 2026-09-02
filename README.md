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

```markdown
# CardioSynth-LSTM 🫀🤖

**Predictive Arrhythmia Forecasting and Synthetic ECG Synthesis using LSTM-Autoencoders and Time-Series GANs (TimeGAN)**

This repository contains a complete biomedical AI framework designed to process physiological time-series data (Electrocardiograms). It addresses data scarcity, class imbalance, and temporal noise in long-term patient monitoring by utilizing deep sequential models. 

This project was developed as a comprehensive open-ended laboratory implementation for the Artificial Intelligence program at Iqra University.

## 📌 Project Overview
The architecture is divided into two primary pipelines and a final evaluation phase:
1. **Anomaly Detection (Path A):** An LSTM-Autoencoder that compresses normal ECG sequences and flags anomalous heartbeats (e.g., Ventricular Premature Contractions) based on high temporal reconstruction loss[cite: 1].
2. **Data Synthesis (Path B):** A Recurrent Time-Series Generative Adversarial Network (TimeGAN) that synthesizes realistic minority-class ECG beats to mitigate training dataset imbalance[cite: 1].
3. **Downstream Classification:** A secondary LSTM classifier trained on augmented (real + synthetic) data, benchmarked against a baseline model trained strictly on imbalanced real data[cite: 1].

## 📂 Repository Structure
The codebase is structured into modular directories as per the project requirements[cite: 1]:

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

```

## 📊 Dataset & Uniqueness Mechanism

* **Source:** PhysioNet MIT-BIH Arrhythmia Database.


* **Preprocessing:** Signals are filtered using a 0.5-45 Hz Butterworth bandpass filter and Z-score normalized.


* **Sequence Windowing:** To ensure uniqueness, a custom sliding window of **$N = 190$ time-steps** (derived from student identification parameters) is applied with a 50% overlap across a 14-patient record subset.



## 🚀 Installation & Usage

**1. Install Dependencies:**

```bash
pip install -r requirements.txt

```

**2. Execute the Pipeline:**

* **Step 1:** Run `data_loader/physionet_ingestion.py` to fetch and preprocess the data.
* **Step 2:** Run `models/LSTM_Autoencoder.py` to train the anomaly detector (Outputs MSE loss and reconstruction graphs).
* **Step 3:** Run `models/TimeGAN.py` to initiate adversarial training for the minority Ventricular class.
* **Step 4:** Run `evaluation/classification_eval.py` to evaluate the downstream LSTM classifier.

## 📈 Experimental Results

* **LSTM-Autoencoder:** Successfully isolated cardiac arrhythmias. Normal rhythms maintained an MSE of ~0.0155, while Ventricular anomalies spiked to ~18.56.
* **Classifier Performance:**
* Baseline Accuracy (Strictly Imbalanced Data): **97.09%**

* Hybrid Accuracy (GAN Augmented Data): **97.25%**



* *Note on TimeGAN:* Early epoch generation exhibited mode collapse (producing static sequences). Full convergence requires extended adversarial training (1,000+ epochs).



---

**Author:** Asif Ali
```
