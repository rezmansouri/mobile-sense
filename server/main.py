#!/usr/bin/env python3
"""
tap_prediction_server.py

TCP server that receives sensor data from Android app and predicts digits using trained model.
Outputs results in format: XXXX or XXX
confidence: X% - X% - X% - X% or X% - X% - X%
"""
import socket
import threading
import logging
import csv
import io
import numpy as np
import torch
import torch.nn as nn
import joblib


# Auto-detect local IP
def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "0.0.0.0"


HOST = get_local_ip()
PORT = 8080

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ConvBlock(nn.Module):
    """Conv1D block: Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBiLSTM(nn.Module):
    """
    Encoder: stack of convolutional blocks with stride-downsampling (no MaxPool)
    BiLSTM: 2-layer bidirectional with dropout, LayerNorm before LSTM
    Classification head: FC layers with dropout
    """

    def __init__(
        self,
        in_channels=9,
        conv_channels=128,
        lstm_hidden=128,
        dropout=0.3,
        num_classes=10,
    ):
        super().__init__()
        # conv encoder: (C=9, T=81) -> downsample twice (T/4 ~ 20)
        self.enc = nn.Sequential(
            ConvBlock(in_channels, conv_channels // 2, kernel_size=7, stride=1),
            ConvBlock(
                conv_channels // 2, conv_channels, kernel_size=5, stride=2
            ),  # downsample x2
            ConvBlock(conv_channels, conv_channels, kernel_size=5, stride=1),
            ConvBlock(
                conv_channels, conv_channels, kernel_size=3, stride=2
            ),  # downsample x2 -> total x4
        )
        self.dropout = nn.Dropout(dropout)

        # LayerNorm applied over channel dimension (after transpose to [B, T, C])
        self.layer_norm = nn.LayerNorm(
            conv_channels
        )  # will initialize lazily when forward sees shape

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.25,
            batch_first=True,
            bidirectional=True,
        )

        fc_input = lstm_hidden * 2  # bidirectional

        self.head = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [B, C=9, T=81]
        x = x.float()
        x = self.enc(x)  # -> [B, C_enc, T_enc]
        # transpose to [B, T_enc, C_enc] for LSTM
        x = x.transpose(1, 2)
        if self.layer_norm is None:
            # initialize LayerNorm to normalize over channel dim (C_enc)
            self.layer_norm = nn.LayerNorm(x.size(-1)).to(x.device)
        x = self.layer_norm(x)
        x, _ = self.lstm(x)  # x: [B, T_enc, 2*lstm_hidden]
        # mean pooling over time
        x = x.mean(dim=1)
        x = self.head(x)
        return x


class TapPredictor:
    def __init__(self, model_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint["hyper"]

        # Create model and load weights
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        print("Tap predictor initialized successfully")

    def _create_model(self):
        model = ConvBiLSTM(
            in_channels=9,
            conv_channels=self.config["conv_channels"],
            lstm_hidden=self.config["lstm_hidden"],
            dropout=self.config["dropout"],
            num_classes=10,
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

        return predicted_digit, confidence


class SensorServer:
    def __init__(self, model_path, scaler_path):
        self.predictor = TapPredictor(model_path, scaler_path)
        self.current_password = []
        self.current_confidences = []
        self.expected_digits = 4  # Default to 4-digit mode

    def parse_sensor_csv(self, csv_line):
        """Parse CSV line containing sensor data."""
        try:
            reader = csv.reader(io.StringIO(csv_line))
            row = next(reader)
            sensor_data = [float(x) for x in row]

            if len(sensor_data) != 729:
                logging.warning(f"Expected 729 values, got {len(sensor_data)}")
                return None

            return sensor_data

        except Exception as e:
            logging.error(f"Failed to parse sensor CSV: {e}")
            return None

    def handle_mode_signal(self, mode_line):
        """Handle mode signal from client (MODE:3 or MODE:4)"""
        try:
            if mode_line.startswith("MODE:"):
                digits = int(mode_line.split(":")[1])
                self.expected_digits = digits
                print(f"Mode changed: Expecting {digits}-digit passcode")
                logging.info(f"Client switched to {digits}-digit mode")
        except Exception as e:
            logging.error(f"Error parsing mode signal: {e}")

    def handle_sensor_data(self, sensor_data):
        """Process received sensor data and predict digit."""
        try:
            # Get prediction
            predicted_digit, confidence = self.predictor.predict(sensor_data)

            # Store digit and confidence
            self.current_password.append(predicted_digit)
            self.current_confidences.append(confidence)

            # Check if we have expected number of digits
            if len(self.current_password) == self.expected_digits:
                # Format the output exactly as requested
                password_str = "".join(str(d) for d in self.current_password)
                confidence_str = "-".join(
                    f"{c*100:.1f}%" for c in self.current_confidences
                )

                print("=" * 30)
                print(f"PREDICTED PASSWORD: {password_str}")
                print(f"CONFIDENCE: {confidence_str}")

                # Reset for next password
                self.current_password = []
                self.current_confidences = []

        except Exception as e:
            logging.error(f"Prediction error: {e}")

    def client_thread(self, conn, addr):
        logging.info("Client connected: %s", addr)

        try:
            conn.send(b"Android Sensor Prediction Server Ready\n")
        except:
            pass

        buffer = b""

        try:
            while True:
                data = conn.recv(8192)
                if not data:
                    logging.info("Client %s disconnected", addr)
                    break

                buffer += data

                # Process complete lines
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line = line_bytes.decode("utf-8").strip()

                    if line:
                        # Check if this is a mode signal
                        if line.startswith("MODE:"):
                            self.handle_mode_signal(line)
                        else:
                            # Otherwise it's sensor data
                            sensor_data = self.parse_sensor_csv(line)
                            if sensor_data is not None:
                                self.handle_sensor_data(sensor_data)

        except ConnectionError:
            logging.info("Connection error with %s — closing", addr)
        except Exception as e:
            logging.exception("Unexpected error with %s: %s", addr, e)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            logging.info("Closed connection: %s", addr)

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()

            logging.info("=" * 50)
            logging.info("Tap Prediction Server Started!")
            logging.info("Server IP: %s", HOST)
            logging.info("Port: %d", PORT)
            logging.info("Model loaded successfully")
            logging.info("=" * 50)
            logging.info("On your Android phone, enter this IP and port")
            logging.info("Waiting for connections...")

            while True:
                conn, addr = s.accept()
                t = threading.Thread(
                    target=self.client_thread, args=(conn, addr), daemon=True
                )
                t.start()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Tap Prediction Server")
    parser.add_argument(
        "--model",
        type=str,
        default="best_model_config_1.pth",
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--scaler", type=str, default="sensor_scaler.pkl", help="Path to scaler file"
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please provide the correct path to your model checkpoint")
        exit(1)

    if not os.path.exists(args.scaler):
        print(f"Scaler file not found: {args.scaler}")
        print("Please provide the correct path to your scaler file")
        exit(1)

    try:
        server = SensorServer(args.model, args.scaler)
        server.start_server()
    except KeyboardInterrupt:
        print("\nServer shutdown by user")
        logging.info("Server shutdown by user")
    except Exception as e:
        print(f"Server error: {e}")
        logging.exception("Server error: %s", e)
