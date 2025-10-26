# test_client.py
import socket, struct, json


def send_array(host="127.0.0.1", port=50000, arr=[1, 2, 3.5]):
    data = json.dumps(arr).encode("utf-8")
    header = struct.pack(">I", len(data))
    with socket.create_connection((host, port)) as s:
        s.sendall(header + data)


if __name__ == "__main__":
    send_array(arr=[100, 20.5, -3.1415])
