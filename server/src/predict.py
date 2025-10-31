import torch
import torch.nn as nn
import numpy as np
import joblib

class TapPredictor:
    def __init__(self, model_path="best_model_config_1.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Create model and load weights
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        self.scaler = joblib.load('sensor_scaler.pkl')
        
        print("Tap predictor initialized successfully")
    
    def _create_model(self):
        model = HybridCNN_LSTM(
            conv_filters=self.config['conv_filters'],
            lstm_units=self.config['lstm_units'],
            dropout_rate=self.config['dropout']
        ).to(self.device)
        return model
    
    def predict(self, sensor_data):
        """
        Predict digit from sensor data
        sensor_data: list of 729 floats (9 sensors × 81 samples)
        Returns: predicted digit (0-9) and confidence scores
        """
        # Convert to numpy and ensure correct shape
        sensor_array = np.array(sensor_data, dtype=np.float32)
        
        if sensor_array.shape != (729,):
            raise ValueError(f"Expected 729 values, got {sensor_array.shape}")
        
        # Apply scaling (same as training)
        sensor_reshaped = sensor_array.reshape(9, 81).transpose(1, 0)  # (81, 9)
        sensor_scaled = self.scaler.transform(sensor_reshaped)
        sensor_final = sensor_scaled.transpose(1, 0).reshape(1, 9, 81)  # (1, 9, 81)
        
        # Convert to tensor
        sensor_tensor = torch.tensor(sensor_final, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sensor_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_digit = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_digit].item()
        
        # Get all probabilities
        all_probs = probabilities.cpu().numpy()[0]
        
        return {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'all_probabilities': all_probs.tolist(),
            'top_3_predictions': self._get_top_predictions(all_probs)
        }
    
    def _get_top_predictions(self, probabilities, top_k=3):
        """Get top K predictions with their probabilities"""
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        return [(int(idx), float(probabilities[idx])) for idx in top_indices]

# Model definition (same as training)
class HybridCNN_LSTM(nn.Module):
    def __init__(self, conv_filters=64, lstm_units=64, dropout_rate=0.3):
        super(HybridCNN_LSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(9, conv_filters, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters//2, kernel_size=10, padding=5)
        self.bn2 = nn.BatchNorm1d(conv_filters//2)
        self.pool = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(conv_filters//2, lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(lstm_units * 2, 64)
        self.fc2 = nn.Linear(64, 10)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        
        x = torch.mean(lstm_out, dim=1)
        x = self.dropout(x)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Usage example
if __name__ == "__main__":
    # Example usage
    predictor = TapPredictor()
    
    # Example: Create dummy sensor data (replace with real data)
    dummy_sensor_data = np.random.randn(729).tolist()
    
    result = predictor.predict(dummy_sensor_data)
    print(f"Predicted digit: {result['predicted_digit']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Top 3 predictions: {result['top_3_predictions']}")