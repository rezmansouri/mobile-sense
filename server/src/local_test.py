#!/usr/bin/env python3
"""
test_sensor_client.py

Test client for sending sensor data to the server.
Sends CSV data: sensor_0_sample_0,sensor_0_sample_1,...,sensor_8_sample_80
Each line contains 729 float values (9 sensors × 81 samples)
"""
import socket
import random
import time

def generate_sensor_data():
    """
    Generate realistic sensor data for testing.
    Returns a list of 729 float values (9 sensors × 81 samples)
    """
    sensor_data = []
    
    # Generate data for each of the 9 sensors
    for sensor in range(9):
        # Each sensor has 81 samples
        for sample in range(81):
            # Generate realistic sensor values based on sensor type
            if sensor < 3:  # Accelerometer (sensors 0-2)
                # Typical accelerometer values with some noise
                base_value = random.uniform(-2.0, 2.0)
                noise = random.gauss(0, 0.1)
                value = base_value + noise
            elif sensor < 6:  # Gyroscope (sensors 3-5)
                # Typical gyroscope values
                base_value = random.uniform(-5.0, 5.0)
                noise = random.gauss(0, 0.05)
                value = base_value + noise
            else:  # Rotation vector (sensors 6-8)
                # Typical rotation values
                base_value = random.uniform(-3.14, 3.14)
                noise = random.gauss(0, 0.02)
                value = base_value + noise
            
            sensor_data.append(value)
    
    return sensor_data

def send_sensor_data(host="127.0.0.1", port=8080, sensor_data=None, num_taps=1, delay=0.5):
    """
    Send sensor data to the server in CSV format.
    
    Args:
        host: Server hostname or IP
        port: Server port
        sensor_data: List of 729 floats, or None to generate random data
        num_taps: Number of tap events to simulate
        delay: Delay between taps in seconds
    """
    if sensor_data is None:
        sensor_data = generate_sensor_data()
    
    # Validate data length
    if len(sensor_data) != 729:
        raise ValueError(f"Expected 729 sensor values, got {len(sensor_data)}")
    
    # Convert to CSV string
    csv_line = ",".join(f"{x:.6f}" for x in sensor_data) + "\n"
    
    print(f"Connecting to {host}:{port}")
    print(f"Sending {num_taps} tap(s) with {len(sensor_data)} sensor values")
    
    with socket.create_connection((host, port), timeout=10) as sock:
        # Receive welcome message (if any)
        try:
            welcome = sock.recv(1024)
            if welcome:
                print(f"Server: {welcome.decode().strip()}")
        except socket.timeout:
            print("No welcome message from server")
        
        for tap_num in range(num_taps):
            # Generate new data for each tap if not provided
            if tap_num > 0:
                if sensor_data is None:
                    current_data = generate_sensor_data()
                else:
                    # Add some variation to the provided data
                    current_data = [x + random.gauss(0, 0.01) for x in sensor_data]
                csv_line = ",".join(f"{x:.6f}" for x in current_data) + "\n"
            
            # Send CSV data
            data_bytes = csv_line.encode('utf-8')
            sock.sendall(data_bytes)
            
            print(f"Sent tap #{tap_num + 1}: {len(data_bytes)} bytes")
            
            # Wait before sending next tap
            if tap_num < num_taps - 1:
                time.sleep(delay)
        
        print("All data sent successfully")

def send_multiple_patterns(host="127.0.0.1", port=8080):
    """
    Send different sensor patterns to test various scenarios.
    """
    patterns = {
        "zeros": [0.0] * 729,
        "ones": [1.0] * 729,
        "sequential": [float(i) for i in range(729)],
        "sin_wave": [float(i % 81) for i in range(729)],  # Repeating pattern every 81 samples
        "random": [random.uniform(-10, 10) for _ in range(729)]
    }
    
    for pattern_name, sensor_data in patterns.items():
        print(f"\n{'='*50}")
        print(f"Sending pattern: {pattern_name}")
        print(f"{'='*50}")
        
        try:
            send_sensor_data(host, port, sensor_data, num_taps=1, delay=0)
            time.sleep(1)  # Brief pause between patterns
        except Exception as e:
            print(f"Error sending {pattern_name}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test sensor data client')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--taps', type=int, default=1, help='Number of taps to send')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between taps')
    parser.add_argument('--pattern', action='store_true', help='Send multiple test patterns')
    
    args = parser.parse_args()
    
    try:
        if args.pattern:
            send_multiple_patterns(args.host, args.port)
        else:
            send_sensor_data(
                host=args.host,
                port=args.port,
                sensor_data=None,  # Generate random data
                num_taps=args.taps,
                delay=args.delay
            )
    except ConnectionRefusedError:
        print(f"Connection refused: Is the server running on {args.host}:{args.port}?")
    except Exception as e:
        print(f"Error: {e}")