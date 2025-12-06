#!/usr/bin/env python3
"""
tcp_sensor_server.py

Runs a threaded TCP server for receiving sensor data from Android app.
Automatically detects local IP address.
"""
import socket
import threading
import logging
import csv
import io
from datetime import datetime
import numpy as np

# Auto-detect local IP
def get_local_ip():
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Doesn't actually send data
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "0.0.0.0"  # Fallback to all interfaces

HOST = get_local_ip()  # Auto-detect local IP
PORT = 8080  # default port

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Statistics
total_taps_received = 0
sensor_data_buffer = []

def handle_sensor_data(client_addr, sensor_data):
    """Process received sensor data."""
    global total_taps_received
    total_taps_received += 1
    
    sensor_array = np.array(sensor_data)
    sensor_matrix = sensor_array.reshape(9, 81)
    
    means = np.mean(sensor_matrix, axis=1)
    stds = np.std(sensor_matrix, axis=1)
    ranges = np.ptp(sensor_matrix, axis=1)
    
    logging.info(f"Tap #{total_taps_received} from {client_addr}")
    logging.info(f"  Sensor means: {[f'{m:.3f}' for m in means]}")
    logging.info(f"  Sensor stds:  {[f'{s:.3f}' for s in stds]}")
    logging.info(f"  Data range:   {[f'{r:.3f}' for r in ranges]}")

def parse_sensor_csv(csv_line):
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

def client_thread(conn, addr):
    logging.info("Client connected: %s", addr)
    
    try:
        conn.send(b"Android Sensor Server Ready\n")
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
            while b'\n' in buffer:
                line_bytes, buffer = buffer.split(b'\n', 1)
                line = line_bytes.decode('utf-8').strip()
                
                if line:
                    sensor_data = parse_sensor_csv(line)
                    if sensor_data is not None:
                        handle_sensor_data(addr, sensor_data)
                        
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

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        
        logging.info("=" * 50)
        logging.info("Sensor Server Started Successfully!")
        logging.info("Server IP: %s", HOST)
        logging.info("Port: %d", PORT)
        logging.info("=" * 50)
        logging.info("On your Android phone, enter this IP and port")
        logging.info("Waiting for connections...")
        
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=client_thread, args=(conn, addr), daemon=True)
            t.start()

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        logging.info("Server shutdown by user")
    except Exception as e:
        logging.exception("Server error: %s", e)