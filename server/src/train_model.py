import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class
class SensorDataset(Dataset):
    def __init__(self, data, labels, scaler=None, fit_scaler=False):
        self.data = data
        self.labels = labels
        
        if fit_scaler and scaler is not None:
            # Reshape for scaling: (samples, 729) -> (samples*9, 81)
            data_reshaped = data.reshape(-1, 9, 81).transpose(0, 2, 1).reshape(-1, 9)
            scaler.fit(data_reshaped)
        
        if scaler is not None:
            # Apply scaling per sensor
            data_reshaped = data.reshape(-1, 9, 81).transpose(0, 2, 1).reshape(-1, 9)
            data_scaled = scaler.transform(data_reshaped)
            # Reshape back
            self.data = data_scaled.reshape(-1, 81, 9).transpose(0, 2, 1).reshape(-1, 729)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Reshape from 729 to (9, 81)
        sensor_data = self.data[idx].reshape(9, 81).astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(sensor_data), torch.tensor(label, dtype=torch.long)

# Hybrid CNN-LSTM Model
class HybridCNN_LSTM(nn.Module):
    def __init__(self, conv_filters=64, lstm_units=64, dropout_rate=0.3):
        super(HybridCNN_LSTM, self).__init__()
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(9, conv_filters, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters//2, kernel_size=10, padding=5)
        self.bn2 = nn.BatchNorm1d(conv_filters//2)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(conv_filters//2, lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier
        self.fc1 = nn.Linear(lstm_units * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 10)  # 10 digits (0-9)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # CNN forward
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # LSTM forward
        x = x.transpose(1, 2)  # (batch, features, time) -> (batch, time, features)
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling over time
        x = torch.mean(lstm_out, dim=1)
        x = self.dropout(x)
        
        # Classifier
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def load_and_preprocess_data(data_folder):
    """Load all CSV files and preprocess data"""
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_folder}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    all_labels = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        print(f"Loading {file}: {len(df)} samples")
        
        # Extract sensor data (columns 3 to 731)
        sensor_data = df.iloc[:, 3:732].values  # 729 sensor values
        labels = df.iloc[:, 0].values  # digit labels
        
        all_data.append(sensor_data)
        all_labels.append(labels)
    
    # Combine all data
    data = np.vstack(all_data)
    labels = np.hstack(all_labels)
    
    print(f"Total samples: {data.shape[0]}")
    print(f"Data shape: {data.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return data, labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    # Configuration
    data_folder = "sensor_data"  # Change this to your data folder
    batch_size = 32
    num_epochs = 30
    
    # Hyperparameter grid
    hyperparam_grid = [
        {'conv_filters': 64, 'lstm_units': 64, 'lr': 0.001, 'dropout': 0.3},
        {'conv_filters': 128, 'lstm_units': 64, 'lr': 0.001, 'dropout': 0.3},
        {'conv_filters': 64, 'lstm_units': 128, 'lr': 0.0005, 'dropout': 0.4},
        {'conv_filters': 96, 'lstm_units': 96, 'lr': 0.001, 'dropout': 0.2}
    ]
    
    # Load and split data
    print("Loading data...")
    data, labels = load_and_preprocess_data(data_folder)
    
    # Split data: 70% train, 20% val, 10% test
    n_total = len(data)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    
    # Create scaler for training data
    scaler = MinMaxScaler()
    
    # Split indices
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Create datasets with proper scaling
    train_dataset = SensorDataset(data[train_idx], labels[train_idx], scaler, fit_scaler=True)
    val_dataset = SensorDataset(data[val_idx], labels[val_idx], scaler, fit_scaler=False)
    test_dataset = SensorDataset(data[test_idx], labels[test_idx], scaler, fit_scaler=False)
    
    # Save test indices for later use
    np.save('test_indices.npy', test_idx)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'sensor_scaler.pkl')
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Train each hyperparameter configuration
    best_val_loss = float('inf')
    best_model_path = None
    best_config = None
    all_results = {}
    
    for i, config in enumerate(hyperparam_grid):
        print(f"\n{'='*50}")
        print(f"Training configuration {i+1}/{len(hyperparam_grid)}")
        print(f"Config: {config}")
        print(f"{'='*50}")
        
        # Create model
        model = HybridCNN_LSTM(
            conv_filters=config['conv_filters'],
            lstm_units=config['lstm_units'],
            dropout_rate=config['dropout']
        ).to(device)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # Training history
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        # Training loop
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f'best_model_config_{i+1}.pth'
                best_config = config
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epoch': epoch
                }, best_model_path)
                print(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Store results
        all_results[f'config_{i+1}'] = {
            'config': config,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'final_val_loss': val_losses[-1],
            'final_val_acc': val_accs[-1]
        }
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'Loss - Config {i+1}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title(f'Accuracy - Config {i+1}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_history_config_{i+1}.png')
        plt.close()
    
    # Save all results
    with open('training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETED")
    print(f"{'='*50}")
    print(f"Best configuration: {best_config}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved as: {best_model_path}")
    
    # Test the best model
    print("\nTesting best model on test set...")
    test_model(best_model_path, test_dataset, device)

def test_model(model_path, test_dataset, device):
    """Test the best model on test set"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = HybridCNN_LSTM(
        conv_filters=config['conv_filters'],
        lstm_units=config['lstm_units'],
        dropout_rate=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    
    print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'config': config
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    main()