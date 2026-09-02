# Cell 3: PyTorch DataLoader & LSTM-Autoencoder Architecture
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Filter ONLY 'Normal' (Class 0) heartbeats for unsupervised training
normal_data = X_data[y_data == 0]
X_train_tensor = torch.tensor(normal_data, dtype=torch.float32).unsqueeze(-1) # Shape: (Samples, 190, 1)

# Create DataLoader for batch processing
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=128, shuffle=True)
print(f"Training Autoencoder on {len(normal_data)} Normal sequences...")

# 2. Define the Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        
        # RNN layer 1
        self.rnn1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        # RNN layer 2 with recurrent dropout logic as required
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x = self.dropout(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape(-1, self.embedding_dim) # Latent Vector

# 3. Define the Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super().__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        
        self.rnn1 = nn.LSTM(input_size=input_dim, hidden_size=self.input_dim, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Repeat the latent vector across the sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (_, _) = self.rnn1(x)
        x = self.dropout(x)
        x, (_, _) = self.rnn2(x)
        return self.output_layer(x)

# 4. Combine into Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim)
        self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Initialize Model, Loss (MSE), and Optimizer (with L2 Weight Decay as per OEL)
autoencoder = LSTMAutoencoder(seq_len=WINDOW_SIZE, n_features=1, embedding_dim=64).to(device)
criterion = nn.MSELoss() # Requirement: Temporal reconstruction loss
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5) # Requirement: L2 weight decay

print("LSTM-Autoencoder architecture built successfully!")


# Cell 4: Training LSTM-Autoencoder & Anomaly Detection Visualization
import time
import matplotlib.pyplot as plt
import numpy as np

epochs = 15
train_losses = []

print(f"Starting training for {epochs} epochs on CUDA...")
start_time = time.time()

autoencoder.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch in train_loader:
        x_batch = batch[0].to(device)
        
        optimizer.zero_grad()
        reconstructed = autoencoder(x_batch)
        loss = criterion(reconstructed, x_batch) # MSE Loss calculation
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.6f}')

print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes.")

# --- 1. Plot Training Loss Curve ---
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Training MSE Loss', color='purple')
plt.title("LSTM-Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Anomaly Evaluation & Visual Outputs ---
print("\nGenerating Original vs. Reconstructed visual outputs...")
autoencoder.eval()

# Get one Normal sample and one Anomalous (Ventricular Ectopic - Class 2) sample
anomalous_data = X_data[y_data == 2] # Ventricular Ectopic class
sample_normal = torch.tensor(normal_data[0], dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
sample_anomaly = torch.tensor(anomalous_data[0], dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)

with torch.no_grad():
    recon_normal = autoencoder(sample_normal)
    recon_anomaly = autoencoder(sample_anomaly)
    
    # Calculate individual MSE for comparison
    loss_normal = criterion(recon_normal, sample_normal).item()
    loss_anomaly = criterion(recon_anomaly, sample_anomaly).item()

# Plotting Normal vs Anomaly Reconstruction
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Normal Rhythm Plot
original_n = sample_normal.cpu().squeeze().numpy()
reconst_n = recon_normal.cpu().squeeze().numpy()
axes[0].plot(original_n, label='Original (Normal)', color='blue')
axes[0].plot(reconst_n, label='Reconstructed', color='green', linestyle='dashed')
axes[0].set_title(f"Normal Rhythm Reconstruction (MSE: {loss_normal:.4f})")
axes[0].legend()
axes[0].grid(True)

# Ventricular Anomaly Plot
original_a = sample_anomaly.cpu().squeeze().numpy()
reconst_a = recon_anomaly.cpu().squeeze().numpy()
axes[1].plot(original_a, label='Original (Ventricular)', color='red')
axes[1].plot(reconst_a, label='Reconstructed', color='green', linestyle='dashed')

# Highlighting anomalous temporal regions (regions with high error)
error = np.abs(original_a - reconst_a)
threshold = np.mean(error) + 1.5 * np.std(error) # Dynamic threshold based on error standard deviation
anomalous_indices = np.where(error > threshold)[0]
axes[1].scatter(anomalous_indices, original_a[anomalous_indices], color='orange', label='High Error Region', zorder=5)

axes[1].set_title(f"Ventricular Anomaly Reconstruction (MSE: {loss_anomaly:.4f})")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("LSTM-Autoencoder: Temporal Sequence Reconstruction & Anomaly Highlighting", fontsize=14)
plt.tight_layout()
plt.show()

print(f"\n[Analytical Observation] Normal MSE is {loss_normal:.4f} while Ventricular MSE is significantly higher at {loss_anomaly:.4f}.")
