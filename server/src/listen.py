#!/usr/bin/env python3
"""
tcp_array_server.py

Runs a threaded TCP server. Protocol:
- Client sends 4-byte big-endian unsigned int (message length N)
- Then N bytes of UTF-8 JSON representing an array of numbers, e.g. [1, 2, 3.5]

Server decodes the JSON and passes it to handle_array(client_addr, array).
"""
import socket
import threading
import struct
import json
import logging

HOST = "0.0.0.0"  # bind to all interfaces; change to '127.0.0.1' for local-only
PORT = 50000  # pick any free port

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def handle_array(client_addr, arr):
    # Replace with your app logic
    logging.info("Received from %s: %s", client_addr, arr)


def recv_exact(sock, nbytes):
    """Receive exactly nbytes from sock or raise ConnectionError on EOF."""
    data = bytearray()
    while len(data) < nbytes:
        packet = sock.recv(nbytes - len(data))
        if not packet:
            raise ConnectionError("socket closed while reading")
        data.extend(packet)
    return bytes(data)


def client_thread(conn, addr):
    logging.info("Client connected: %s", addr)
    try:
        while True:
            # Read length prefix (4 bytes)
            header = conn.recv(4)
            if not header:
                logging.info("Client %s disconnected", addr)
                break
            if len(header) < 4:
                # try to read remainder
                header += recv_exact(conn, 4 - len(header))
            (msg_len,) = struct.unpack(">I", header)  # big-endian unsigned int
            if msg_len == 0:
                logging.warning("Received zero-length message from %s", addr)
                continue
            payload = recv_exact(conn, msg_len)
            try:
                text = payload.decode("utf-8")
                arr = json.loads(text)
                # basic validation: must be list of numbers
                if not isinstance(arr, list):
                    logging.warning("Received non-list from %s: %r", addr, arr)
                    continue
                # optionally convert elements to float
                arr = [float(x) for x in arr]
            except Exception as e:
                logging.exception("Failed to decode JSON from %s: %s", addr, e)
                continue
            handle_array(addr, arr)
    except ConnectionError:
        logging.info("Connection error with %s — closing", addr)
    except Exception:
        logging.exception("Unexpected error with %s", addr)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        logging.info("Closed connection: %s", addr)


def start_server(host=HOST, port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        logging.info("Server listening on %s:%d", host, port)
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=client_thread, args=(conn, addr), daemon=True)
            t.start()


if __name__ == "__main__":
    start_server()
