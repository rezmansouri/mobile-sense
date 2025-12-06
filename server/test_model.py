import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import joblib
from train_model import HybridCNN_LSTM, SensorDataset, load_and_preprocess_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load best model
    model_path = "best_model_config_1.pth"  # Change to your best model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Loaded model configuration: {config}")
    
    # Load data and scaler
    data_folder = "sensor_data"  # Same as training
    data, labels = load_and_preprocess_data(data_folder)
    scaler = joblib.load('sensor_scaler.pkl')
    
    # Load test indices
    test_idx = np.load('test_indices.npy')
    test_dataset = SensorDataset(data[test_idx], labels[test_idx], scaler, fit_scaler=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model and load weights
    model = HybridCNN_LSTM(
        conv_filters=config['conv_filters'],
        lstm_units=config['lstm_units'],
        dropout_rate=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test the model
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, labels_batch in test_loader:
            data, labels_batch = data.to(device), labels_batch.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, digits=4))
    
    # Per-digit accuracy
    digit_accuracy = []
    for digit in range(10):
        mask = all_labels == digit
        if np.sum(mask) > 0:
            acc = np.mean(all_predictions[mask] == digit)
            digit_accuracy.append(acc)
            print(f"Digit {digit}: {acc:.4f} ({np.sum(mask)} samples)")
    
    # Save detailed results
    results = {
        'overall_accuracy': accuracy,
        'per_digit_accuracy': digit_accuracy,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions.tolist(),
        'true_labels': all_labels.tolist(),
        'probabilities': all_probabilities.tolist()
    }
    
    import json
    with open('detailed_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to detailed_test_results.json")
    print(f"Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    main()