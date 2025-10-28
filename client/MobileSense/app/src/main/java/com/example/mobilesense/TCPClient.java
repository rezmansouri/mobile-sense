package com.example.mobilesense;

import android.util.Log;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class TCPClient {
    private static final String TAG = "TCPClient";
    private Socket socket;
    private OutputStream outputStream;
    private ExecutorService executor;
    private String serverIp;
    private int serverPort;
    private boolean isConnected = false;

    public TCPClient(String serverIp, int serverPort) {
        this.serverIp = serverIp;
        this.serverPort = serverPort;
        this.executor = Executors.newSingleThreadExecutor();
    }

    public void connect(ConnectionCallback callback) {
        executor.execute(() -> {
            try {
                socket = new Socket(serverIp, serverPort);
                outputStream = socket.getOutputStream();
                isConnected = true;
                Log.d(TAG, "Connected to server " + serverIp + ":" + serverPort);

                if (callback != null) {
                    callback.onConnected();
                }
            } catch (IOException e) {
                Log.e(TAG, "Connection failed: " + e.getMessage());
                isConnected = false;
                if (callback != null) {
                    callback.onError(e.getMessage());
                }
            }
        });
    }

    public void sendSensorData(float[][] sensorData) {
        if (!isConnected || outputStream == null) {
            Log.w(TAG, "Not connected, cannot send data");
            return;
        }

        executor.execute(() -> {
            try {
                // Convert sensor data to CSV format (no digit, no timestamp)
                StringBuilder dataBuilder = new StringBuilder();

                // Add all sensor samples in one row
                for (int sensor = 0; sensor < 9; sensor++) {
                    for (int sample = 0; sample < sensorData[sensor].length; sample++) {
                        if (sensor > 0 || sample > 0) {
                            dataBuilder.append(",");
                        }
                        dataBuilder.append(sensorData[sensor][sample]);
                    }
                }
                dataBuilder.append("\n"); // Newline as delimiter

                byte[] dataBytes = dataBuilder.toString().getBytes(StandardCharsets.UTF_8);
                outputStream.write(dataBytes);
                outputStream.flush();

                Log.d(TAG, "Sent " + dataBytes.length + " bytes to server");

            } catch (IOException e) {
                Log.e(TAG, "Failed to send data: " + e.getMessage());
                isConnected = false;
            }
        });
    }

    public void disconnect() {
        executor.execute(() -> {
            try {
                if (outputStream != null) {
                    outputStream.close();
                }
                if (socket != null) {
                    socket.close();
                }
                isConnected = false;
                Log.d(TAG, "Disconnected from server");
            } catch (IOException e) {
                Log.e(TAG, "Error during disconnect: " + e.getMessage());
            }
        });
    }

    public boolean isConnected() {
        return isConnected;
    }

    public interface ConnectionCallback {
        void onConnected();
        void onError(String error);
    }
}