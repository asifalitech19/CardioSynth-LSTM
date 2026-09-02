
class RecurrentGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, seq_len, output_dim):
        super().__init__()
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        
        # Recurrent Generator as required by OEL
        self.lstm = nn.LSTM(noise_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z shape: (batch_size, seq_len, noise_dim)
        lstm_out, _ = self.lstm(z)
        generated_seq = self.linear(lstm_out)
        return generated_seq

class RecurrentDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Recurrent Discriminator 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        # We take the hidden state of the last layer to make a single real/fake decision
        decision = self.linear(hidden[-1])
        return decision

# Initialize Dimensions and Models
NOISE_DIM = 10
HIDDEN_DIM = 64

generator = RecurrentGenerator(noise_dim=NOISE_DIM, hidden_dim=HIDDEN_DIM, seq_len=WINDOW_SIZE, output_dim=1).to(device)
discriminator = RecurrentDiscriminator(input_dim=1, hidden_dim=HIDDEN_DIM).to(device)

# Using RMSprop for Wasserstein GAN training stability
opt_G = torch.optim.RMSprop(generator.parameters(), lr=0.0002)
opt_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.0002)

print("TimeGAN Architecture (Recurrent Generator & Discriminator) initialized successfully!")






# Cell 6: TimeGAN Adversarial Training (Wasserstein Loss)
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

# We will augment the 'Ventricular' (Class 2) dataset which had ~2833 samples
minority_data = X_data[y_data == 2] 
gan_dataset = torch.tensor(minority_data, dtype=torch.float32).unsqueeze(-1)
gan_loader = DataLoader(TensorDataset(gan_dataset), batch_size=64, shuffle=True)

gan_epochs = 50 # Using 50 for testing, Lab requirement suggests 1000+ for final output
d_losses = []
g_losses = []

print(f"Starting TimeGAN Adversarial Training on {len(minority_data)} sequences for {gan_epochs} epochs...")

generator.train()
discriminator.train()

for epoch in range(gan_epochs):
    for batch in gan_loader:
        real_seqs = batch[0].to(device)
        b_size = real_seqs.size(0)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        opt_D.zero_grad()
        
        # Generate fake sequences
        noise = torch.randn(b_size, WINDOW_SIZE, NOISE_DIM).to(device)
        fake_seqs = generator(noise).detach()
        
        # Wasserstein Loss for D: -mean(D(real)) + mean(D(fake))
        loss_D = -torch.mean(discriminator(real_seqs)) + torch.mean(discriminator(fake_seqs))
        loss_D.backward()
        opt_D.step()
        
        # Weight clipping for Discriminator stability
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
            
        # -----------------
        #  Train Generator
        # -----------------
        opt_G.zero_grad()
        
        noise = torch.randn(b_size, WINDOW_SIZE, NOISE_DIM).to(device)
        generated_seqs = generator(noise)
        
        # Wasserstein Loss for G: -mean(D(fake))
        loss_G = -torch.mean(discriminator(generated_seqs))
        loss_G.backward()
        opt_G.step()
        
    d_losses.append(loss_D.item())
    g_losses.append(loss_G.item())
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{gan_epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

print("\nTimeGAN Training Complete!")

# --- 1. Visual Output: Generator and Discriminator Loss Curves ---
plt.figure(figsize=(10, 4))
plt.plot(d_losses, label='Discriminator Loss (Wasserstein)', color='red')
plt.plot(g_losses, label='Generator Loss (Wasserstein)', color='blue')
plt.title("TimeGAN Adversarial Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Visual Output: Real vs Synthesized ECG ---
generator.eval()
with torch.no_grad():
    sample_noise = torch.randn(1, WINDOW_SIZE, NOISE_DIM).to(device)
    synthetic_beat = generator(sample_noise).cpu().squeeze().numpy()

plt.figure(figsize=(10, 4))
plt.plot(minority_data[0], label='Real Ventricular ECG', color='red')
plt.plot(synthetic_beat, label='TimeGAN Synthesized ECG', color='orange', linestyle='dashed', linewidth=2)
plt.title("Real ECG Heartbeat vs. TimeGAN Synthesized ECG Heartbeat")
plt.xlabel("Time-steps")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show(
    
)
