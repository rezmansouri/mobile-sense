# Mobile-Sense: Smartphone PIN Inference via Motion Sensors

A research project demonstrating keystroke inference through smartphone motion sensor side-channels. This system captures accelerometer, gyroscope, and rotation data during PIN entry and predicts digits using a hybrid CNN-LSTM neural network.

---

<img src="https://github.com/rezmansouri/mobile-sense/blob/main/images/confusion_matrix.png">


## Data Format

CSV files should contain:

- **Column 0**: Digit label (0–9)  
- **Columns 1–2**: Timestamp (optional)  
- **Columns 3–731**: 729 sensor values (9 channels × 81 time steps)

Format:
```
digit,timestamp1,timestamp2,sensor1_t1,sensor1_t2,...,sensor9_t81
```

<img src="https://github.com/rezmansouri/mobile-sense/blob/main/images/sensor_data_visualization.png">

---

## Project Structure
```
mobile-sense/
├── README.md
├── requirements.txt
├── server/
│   ├── main.py                    # Main TCP prediction server
│   ├── train_model.py             # Model training script
│   ├── test_model.py              # Model testing/validation
│   ├── connection_test/
│   │   ├── listen.py              # Simple TCP listener for testing
│   │   └── local_test.py          # Local connection testing
├── results/
│   ├── final/                     # Latest trained models
│   │   ├── best_model.pth         # Trained model weights
│   │   ├── sensor_scaler.pkl      # Fitted StandardScaler
│   └── example/                   # One xample pre-trained model
├── data/                          # Sensor data collection (CSV files)
└── client/                        # Android applications project roots
└── apks/                          # Android applications executables

---

## Quick Start for Server

### 1. Install Dependencies
```bash
cd server
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py ../tap-data/
```
- Trains on CSV files in the specified directory  
- Saves model to `results/final/best_model.pth`  
- Saves scaler to `results/final/sensor_scaler.pkl`  

### 3. Run the Prediction Server
```bash
python src/main.py --model model_path.pth --scaler scaler_path.pkl
```
- Server auto-detects IP (displayed in console)  
- Listens on port **8080**  
- Ready for Android app connections  

### 4. Connect Android App
- Enter server **IP:port** in Android app  
- Start tapping digits  
- Server outputs predictions in real time  

---

### Model Testing
```bash
# Test model performance
python src/test_model.py results/final/best_model.pth data/test-data/
```

---

## Model Architecture

The system uses a **ConvBiLSTM hybrid neural network**:

- **Input:** 9 sensor channels × 81 time steps  
- **Conv Encoder:** 4 convolutional layers with stride downsampling  
- **Temporal Modeling:** 2-layer bidirectional LSTM  
- **Output:** 10-class softmax for digit classification  

---

## Network Protocol

### Android → Server
```
MODE:4
digit,s1,s2,...,s729\n
```

### Server Output
```
PREDICTED PASSWORD: 1234
CONFIDENCE: 85.2% - 72.1% - 91.3% - 68.4%
```

---

## Requirements

See `requirements.txt` for complete list:

- PyTorch  
- scikit-learn  
- NumPy / Pandas  
- Matplotlib  
- joblib  

---

## Testing

### Connection Testing
```bash
cd src/connection_test
python listen.py        # Simple echo server
python local_test.py    # Test data transmission
```

### Performance Metrics
- Inference latency: **~8.7 ms** per sample  
- Server throughput: **115 predictions/sec**  
- Model accuracy: **64.07%** digit classification  

---

## Troubleshooting

- **Connection refused:** Ensure server IP matches Android app  
- **Model not found:** Check paths or use `--model` flag  
- **Memory issues:** Reduce batch size in `train_model.py`  
- **No predictions:** Verify CSV format (81×9 window)  

---

## Research Implications

This project demonstrates:

- Motion sensor data contains keystroke-discriminative information  
- 64% digit classification accuracy achievable with standard sensors  
- Real-time inference feasible on mobile devices  
- Significant privacy implications for sensor-permissioned apps  
