#!/usr/bin/env python3
"""
tcp_sensor_server.py

Runs a threaded TCP server for receiving sensor data from Android app.
Protocol:
- Client sends CSV data: sensor_0_sample_0,sensor_0_sample_1,...,sensor_8_sample_80
- Each line contains 729 float values (9 sensors × 81 samples)
- Lines are newline-delimited
"""
import socket
import threading
import logging
import csv
import io
from datetime import datetime
import numpy as np

HOST = "192.168.11.23" #"0.0.0.0"  # bind to all interfaces
PORT = 8080  # default port matching the Android app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Statistics
total_taps_received = 0
sensor_data_buffer = []  # Buffer to store recent sensor data for analysis

def handle_sensor_data(client_addr, sensor_data):
    """
    Process received sensor data.
    sensor_data: list of 729 floats representing 9 sensors × 81 samples
    """
    global total_taps_received
    
    total_taps_received += 1
    
    # Convert to numpy array for easier manipulation
    sensor_array = np.array(sensor_data)
    
    # Reshape to 9×81 matrix (9 sensors, 81 samples each)
    sensor_matrix = sensor_array.reshape(9, 81)
    
    # Calculate some basic statistics
    means = np.mean(sensor_matrix, axis=1)
    stds = np.std(sensor_matrix, axis=1)
    ranges = np.ptp(sensor_matrix, axis=1)  # peak-to-peak (max-min)
    
    logging.info(f"Tap #{total_taps_received} from {client_addr}")
    logging.info(f"  Sensor means: {[f'{m:.3f}' for m in means]}")
    logging.info(f"  Sensor stds:  {[f'{s:.3f}' for s in stds]}")
    logging.info(f"  Data range:   {[f'{r:.3f}' for r in ranges]}")
    
    # Store in buffer for later analysis
    sensor_data_buffer.append({
        'timestamp': datetime.now(),
        'client_addr': client_addr,
        'sensor_data': sensor_matrix,
        'statistics': {
            'means': means,
            'stds': stds,
            'ranges': ranges
        }
    })
    
    # Keep only last 1000 taps in buffer to prevent memory issues
    if len(sensor_data_buffer) > 1000:
        sensor_data_buffer.pop(0)

def parse_sensor_csv(csv_line):
    """
    Parse CSV line containing sensor data.
    Expected format: 729 comma-separated float values
    """
    try:
        # Parse CSV line
        reader = csv.reader(io.StringIO(csv_line))
        row = next(reader)
        
        # Convert to floats
        sensor_data = [float(x) for x in row]
        
        # Validate length
        if len(sensor_data) != 729:
            logging.warning(f"Expected 729 values, got {len(sensor_data)}")
            return None
            
        return sensor_data
        
    except Exception as e:
        logging.error(f"Failed to parse sensor CSV: {e}")
        logging.debug(f"Problematic data: {csv_line[:100]}...")  # log first 100 chars
        return None

def recv_until_newline(sock, buffer_size=8192):
    """
    Receive data until a newline character is found.
    Returns complete lines and any remaining partial data.
    """
    data = b""
    while True:
        try:
            chunk = sock.recv(buffer_size)
            if not chunk:
                # Connection closed
                return [], data, True
            data += chunk
            if b'\n' in data:
                lines = data.split(b'\n')
                # Return all complete lines and the remaining partial data
                complete_lines = lines[:-1]
                remaining = lines[-1]
                return complete_lines, remaining, False
        except socket.timeout:
            # No data available yet
            if data:
                # Return what we have so far as a partial line
                return [], data, False
            else:
                # No data at all
                return [], b"", False
        except Exception as e:
            logging.error(f"Error receiving data: {e}")
            return [], data, True

def client_thread(conn, addr):
    logging.info("Client connected: %s", addr)
    
    # Set a timeout to prevent blocking forever
    conn.settimeout(1.0)
    
    # Send welcome message (optional)
    try:
        conn.send(b"Android Sensor Server Ready\n")
    except:
        pass
    
    buffer = b""
    
    try:
        while True:
            # Receive data until we get a complete line
            lines, buffer, eof = recv_until_newline(conn)
            
            if eof:
                logging.info("Client %s disconnected", addr)
                break
                
            # Process complete lines
            for line_bytes in lines:
                if not line_bytes:
                    continue
                    
                try:
                    line = line_bytes.decode('utf-8').strip()
                    if not line:
                        continue
                        
                    # Parse sensor data from CSV
                    sensor_data = parse_sensor_csv(line)
                    
                    if sensor_data is not None:
                        handle_sensor_data(addr, sensor_data)
                    else:
                        logging.warning(f"Invalid sensor data from {addr}")
                        
                except UnicodeDecodeError:
                    logging.warning(f"Invalid UTF-8 data from {addr}")
                except Exception as e:
                    logging.error(f"Error processing data from {addr}: {e}")
            
            # If we have partial data in buffer but no complete lines, continue
            if not lines and buffer:
                continue
                    
    except socket.timeout:
        # Timeout is normal - just continue the loop
        pass
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

def print_statistics():
    """Print periodic statistics about received data"""
    global total_taps_received
    if total_taps_received > 0:
        logging.info("=" * 50)
        logging.info(f"STATISTICS: {total_taps_received} total taps received")
        logging.info(f"ACTIVE CLIENTS: {threading.active_count() - 2}")  # subtract main and stats thread
        if sensor_data_buffer:
            latest = sensor_data_buffer[-1]
            logging.info(f"LATEST TAP: {latest['timestamp'].strftime('%H:%M:%S')} from {latest['client_addr']}")
        logging.info("=" * 50)

def statistics_thread():
    """Thread to periodically print server statistics"""
    while True:
        threading.Event().wait(30)  # Print stats every 30 seconds
        print_statistics()

def start_server(host=HOST, port=PORT):
    # Start statistics thread
    stats_thread = threading.Thread(target=statistics_thread, daemon=True)
    stats_thread.start()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        logging.info("Sensor Server listening on %s:%d", host, port)
        logging.info("Waiting for Android client connections...")
        logging.info("Expected data format: 729 comma-separated floats per line")
        logging.info("(9 sensors × 81 samples = 729 values)")
        
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=client_thread, args=(conn, addr), daemon=True)
            t.start()

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        logging.info("Server shutdown by user")
        print_statistics()
    except Exception as e:
        logging.exception("Server error: %s", e)