# Cell 2: Real Data Ingestion & Preprocessing (Roll No: 019)
import wfdb
import numpy as np
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# AAMI ECG Classes Mapping (5 Classes required by OEL)
aami_mapping = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # Supraventricular Ectopic
    'V': 2, 'E': 2,                         # Ventricular Ectopic
    'F': 3,                                 # Fusion
    '/': 4, 'f': 4, 'Q': 4                  # Unknown
}

# Uniqueness Mechanism applied for Roll Number 019
patient_ids = ['100', '101', '105', '111', '113', '118', '119',
               '200', '202', '210', '212', '219', '221', '233']

# Window size N = 190 time-steps (derived from 019)
WINDOW_SIZE = 190
left_window = WINDOW_SIZE // 2
right_window = WINDOW_SIZE - left_window

all_heartbeats = []
all_labels = []

print("Downloading real MIT-BIH records and processing sequences...")

for pid in patient_ids:
    try:
        # Load Record and Annotations via PhysioNet
        record = wfdb.rdrecord(pid, pn_dir='mitdb')
        annotation = wfdb.rdann(pid, 'atr', pn_dir='mitdb')

        # Take the first channel (MLII)
        signal = record.p_signal[:, 0]

        # 1. Bandpass Filtering (0.5 - 45 Hz)
        filtered_signal = nk.signal_filter(signal, sampling_rate=360, lowcut=0.5, highcut=45.0, method='butterworth')

        # 2. Z-score Normalization
        scaler = StandardScaler()
        normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()

        # Extract heartbeats based on R-peaks
        r_peaks = annotation.sample
        symbols = annotation.symbol

        for i in range(len(r_peaks)):
            peak = r_peaks[i]
            sym = symbols[i]

            # Extract window if it fits within bounds and belongs to our 5 classes
            if sym in aami_mapping and (peak - left_window >= 0) and (peak + right_window < len(normalized_signal)):
                beat = normalized_signal[peak - left_window : peak + right_window]
                all_heartbeats.append(beat)
                all_labels.append(aami_mapping[sym])

    except Exception as e:
        print(f"Error processing record {pid}: {e}")

X_data = np.array(all_heartbeats)
y_data = np.array(all_labels)

print(f"\n[Success] Total real heartbeats extracted: {X_data.shape[0]}")
print(f"Shape of each heartbeat sequence: {X_data.shape[1]} time-steps")

# Display class distribution to show imbalance
unique, counts = np.unique(y_data, return_counts=True)
class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
print("\nClass distribution (Notice the heavy imbalance):")
for u, c in zip(unique, counts):
    print(f"- {class_names[u]}: {c} samples")

# Plot a real extracted sequence
plt.figure(figsize=(10, 4))
plt.plot(X_data[0], color='blue')
plt.title(f"Real Normalized ECG Heartbeat (Class: {class_names[y_data[0]]}) - Window: {WINDOW_SIZE}")
plt.xlabel("Time-steps")
plt.ylabel("Z-score Normalized Amplitude")
plt.grid(True)
plt.show()
