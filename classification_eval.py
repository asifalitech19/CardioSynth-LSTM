# Cell 7: Downstream LSTM Classifier (Baseline vs Augmented)
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time

# 1. Prepare Imbalanced Data (Baseline)
# Mapping our 5 classes to 0-4 labels
X_imbalanced = X_data
y_imbalanced = y_data

X_train_imb, X_test, y_train_imb, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.2, random_state=42)

train_tensor_imb = torch.tensor(X_train_imb, dtype=torch.float32).unsqueeze(-1)
y_train_tensor_imb = torch.tensor(y_train_imb, dtype=torch.long)
test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader_imb = DataLoader(TensorDataset(train_tensor_imb, y_train_tensor_imb), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(test_tensor, y_test_tensor), batch_size=128, shuffle=False)

# 2. Prepare Augmented Data (Real + Synthetic)
print("Generating synthetic samples to augment minority class (Ventricular)...")
generator.eval()
num_synthetic_samples = 2000 # Augmenting to balance the Ventricular class slightly
with torch.no_grad():
    noise = torch.randn(num_synthetic_samples, WINDOW_SIZE, NOISE_DIM).to(device)
    synthetic_samples = generator(noise).cpu().squeeze().numpy()
    synthetic_labels = np.full((num_synthetic_samples,), 2) # Label 2 for Ventricular

X_train_aug = np.concatenate((X_train_imb, synthetic_samples), axis=0)
y_train_aug = np.concatenate((y_train_imb, synthetic_labels), axis=0)

train_tensor_aug = torch.tensor(X_train_aug, dtype=torch.float32).unsqueeze(-1)
y_train_tensor_aug = torch.tensor(y_train_aug, dtype=torch.long)
train_loader_aug = DataLoader(TensorDataset(train_tensor_aug, y_train_tensor_aug), batch_size=128, shuffle=True)

# 3. Downstream Classifier Architecture
class DownstreamClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

def train_and_evaluate(train_loader, title="Model"):
    classifier = DownstreamClassifier(input_dim=1, hidden_dim=64, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    
    epochs = 15 # Set low for quick testing, increase for final run
    print(f"\nTraining {title} on {len(train_loader.dataset)} samples...")
    start_time = time.time()
    
    classifier.train()
    for epoch in range(epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    print(f"Training finished in {(time.time() - start_time):.2f} seconds.")
    
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = classifier(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"{title} Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 4. Compare Models as per OEL requirement
baseline_acc = train_and_evaluate(train_loader_imb, "Baseline Model (Strictly Imbalanced Real Data)")
augmented_acc = train_and_evaluate(train_loader_aug, "Hybrid Model (GAN Augmented Data)")

print("\n--- Final Results for Comparison Table ---")
print(f"Classification Accuracy (Imbalanced Data): {baseline_acc:.2f}%")
print(f"Classification Accuracy (GAN Augmented Data): {augmented_acc:.2f}%")
