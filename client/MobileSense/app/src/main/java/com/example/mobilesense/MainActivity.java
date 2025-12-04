package com.example.mobilesense;

import android.graphics.drawable.GradientDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import androidx.appcompat.widget.SwitchCompat;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorEventListener, ServerConfigDialog.ServerConfigListener {
    private LinearLayout pinDots;
    private List<Integer> enteredPin = new ArrayList<>();
    private int MAX_PIN_LENGTH = 4; // Make this variable, not constant

    // Add toggle
    private SwitchCompat pinLengthToggle;
    private boolean is4DigitMode = true;

    // Sensor related variables
    private SensorManager sensorManager;
    private Sensor accelerometer, gyroscope, rotationVector;
    private static final int WINDOW_SIZE = 81;
    private static final int BUFFER_SIZE = 2000;

    // Circular buffer for sensor data
    private float[][] sensorBuffer = new float[9][BUFFER_SIZE];
    private int bufferIndex = 0;
    private boolean isBufferFilled = false;
    private int totalSamplesReceived = 0;

    // TCP Client
    private TCPClient tcpClient;
    private String serverIp = "";
    private int serverPort = 0;

    // UI Elements
    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d("SensorClient", "onCreate started");

        try {
            setContentView(R.layout.activity_main);
            Log.d("SensorClient", "setContentView completed");
        } catch (Exception e) {
            Log.e("SensorClient", "Error in setContentView", e);
            throw e;
        }

        initializeViews();
        Log.d("SensorClient", "initializeViews completed");

        createPinDots();
        Log.d("SensorClient", "createPinDots completed");

        setupButtonClicks();
        Log.d("SensorClient", "setupButtonClicks completed");

        setupToggle();
        Log.d("SensorClient", "setupToggle completed");

        // Start sensors immediately
        setupSensors();
        Log.d("SensorClient", "setupSensors completed");

        // Show server config dialog
        showServerConfigDialog();
        Log.d("SensorClient", "showServerConfigDialog completed");
    }

    private void initializeViews() {
        pinDots = findViewById(R.id.pinDots);
        statusText = findViewById(R.id.statusText);
        pinLengthToggle = findViewById(R.id.pinLengthToggle);
    }

    private void setupToggle() {
        pinLengthToggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                is4DigitMode = isChecked;
                if (is4DigitMode) {
                    MAX_PIN_LENGTH = 4;
                    pinLengthToggle.setText("4-digit PIN");
                } else {
                    MAX_PIN_LENGTH = 3;
                    pinLengthToggle.setText("3-digit PIN");
                }

                // Send mode to server
                sendModeToServer();

                // Clear current PIN and recreate dots
                enteredPin.clear();
                createPinDots();

                updateStatus(is4DigitMode ? "Switched to 4-digit mode" : "Switched to 3-digit mode");
            }
        });
    }

    private void sendModeToServer() {
        if (tcpClient != null && tcpClient.isConnected()) {
            String modeMessage = is4DigitMode ? "MODE:4" : "MODE:3";
            tcpClient.sendMode(modeMessage);
            Log.d("SensorClient", "Sent mode to server: " + modeMessage);
        }
    }

    private void showServerConfigDialog() {
        ServerConfigDialog dialog = new ServerConfigDialog(this, this);
        dialog.show();
        updateStatus("Please configure server connection");
    }

    @Override
    public void onServerConfig(String ip, int port) {
        this.serverIp = ip;
        this.serverPort = port;

        updateStatus("Connecting to " + ip + ":" + port + "...");

        // Initialize TCP client and connect
        tcpClient = new TCPClient(ip, port);
        tcpClient.connect(new TCPClient.ConnectionCallback() {
            @Override
            public void onConnected() {
                runOnUiThread(() -> {
                    updateStatus("Connected to server - Ready for taps");
                    Toast.makeText(MainActivity.this, "Connected to server!", Toast.LENGTH_SHORT).show();
                    // Send initial mode
                    sendModeToServer();
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    updateStatus("Connection failed: " + error);
                    Toast.makeText(MainActivity.this, "Connection failed: " + error, Toast.LENGTH_LONG).show();
                    // Retry connection
                    showServerConfigDialog();
                });
            }
        });
    }

    private void setupSensors() {
        try {
            sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

            if (sensorManager != null) {
                accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
                gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
                rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

                if (accelerometer != null) {
                    sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
                }
                if (gyroscope != null) {
                    sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME);
                }
                if (rotationVector != null) {
                    sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_GAME);
                }

                updateStatus("Sensors initialized - Waiting for connection...");
            }
        } catch (Exception e) {
            Log.e("SensorClient", "Error setting up sensors", e);
            updateStatus("Sensor error: " + e.getMessage());
        }
    }

    private void setupButtonClicks() {
        int[] buttonNumbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};

        for (int number : buttonNumbers) {
            String buttonId = "button_" + number;
            int resId = getResources().getIdentifier(buttonId, "id", getPackageName());
            GlassButton button = findViewById(resId);

            if (button != null) {
                button.setButtonNumber(number);
                button.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        GlassButton glassButton = (GlassButton) v;
                        int digit = glassButton.getButtonNumber();
                        Log.d("SensorClient", "Button pressed: " + digit);
                        onNumberPressed(digit);
                    }
                });
            }
        }
    }

    private void createPinDots() {
        pinDots.removeAllViews();

        for (int i = 0; i < MAX_PIN_LENGTH; i++) {
            View dot = new View(this);
            int size = getResources().getDimensionPixelSize(R.dimen.pin_dot_size);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(size, size);
            params.setMargins(32, 0, 32, 0);
            dot.setLayoutParams(params);

            GradientDrawable dotDrawable = new GradientDrawable();
            dotDrawable.setShape(GradientDrawable.OVAL);
            dotDrawable.setColor(0x60FFFFFF);
            dot.setBackground(dotDrawable);

            pinDots.addView(dot);
        }
        updatePinDots();
    }

    private void onNumberPressed(int number) {
        if (enteredPin.size() < MAX_PIN_LENGTH) {
            enteredPin.add(number);
            updatePinDots();

            Log.d("SensorClient", "Capturing data for digit: " + number +
                    ", Buffer filled: " + isBufferFilled +
                    ", Total samples: " + totalSamplesReceived);

            // Capture and send sensor data for this tap
            captureAndSendSensorData();

            if (enteredPin.size() == MAX_PIN_LENGTH) {
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        clearPin();
                    }
                }, 300);
            }
        }
    }

    private void captureAndSendSensorData() {
        if (!isBufferFilled) {
            Log.w("SensorClient", "Buffer not filled yet! Only have " + totalSamplesReceived + " samples");
            Toast.makeText(this, "Waiting for sensor data...", Toast.LENGTH_SHORT).show();
            return;
        }

        if (tcpClient == null || !tcpClient.isConnected()) {
            Log.w("SensorClient", "Not connected to server");
            Toast.makeText(this, "Not connected to server", Toast.LENGTH_SHORT).show();
            return;
        }

        float[][] tapData = extractSensorWindow();

        if (tapData != null) {
            // Send data via TCP
            tcpClient.sendSensorData(tapData);
            Log.d("SensorClient", "Sensor data sent to server");

            runOnUiThread(() -> {
                updateStatus("Data sent to server");
            });
        } else {
            Log.e("SensorClient", "Failed to extract sensor window");
        }
    }

    private float[][] extractSensorWindow() {
        int startIndex = (bufferIndex - WINDOW_SIZE + BUFFER_SIZE) % BUFFER_SIZE;
        float[][] window = new float[9][WINDOW_SIZE];

        for (int i = 0; i < WINDOW_SIZE; i++) {
            int dataIndex = (startIndex + i) % BUFFER_SIZE;
            for (int sensor = 0; sensor < 9; sensor++) {
                window[sensor][i] = sensorBuffer[sensor][dataIndex];
            }
        }

        // Verify we have valid data
        boolean hasValidData = false;
        for (int sensor = 0; sensor < 9; sensor++) {
            for (int sample = 0; sample < WINDOW_SIZE; sample++) {
                if (window[sensor][sample] != 0.0f) {
                    hasValidData = true;
                    break;
                }
            }
            if (hasValidData) break;
        }

        if (!hasValidData) {
            Log.w("SensorClient", "Extracted window contains only zeros!");
            return null;
        }

        return window;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        totalSamplesReceived++;

        int sensorType = event.sensor.getType();
        int baseIndex = -1;

        if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            baseIndex = 0;
        } else if (sensorType == Sensor.TYPE_GYROSCOPE) {
            baseIndex = 3;
        } else if (sensorType == Sensor.TYPE_ROTATION_VECTOR) {
            baseIndex = 6;
        } else {
            return;
        }

        // Store sensor data
        for (int i = 0; i < 3 && i < event.values.length; i++) {
            sensorBuffer[baseIndex + i][bufferIndex] = event.values[i];
        }

        bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;

        // Set buffer as filled after we have enough samples
        if (totalSamplesReceived >= WINDOW_SIZE && !isBufferFilled) {
            isBufferFilled = true;
            Log.d("SensorClient", "*** BUFFER NOW FILLED! *** Total samples: " + totalSamplesReceived);

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    updateStatus("Ready - Sensors active (" + totalSamplesReceived + " samples)");
                    Toast.makeText(MainActivity.this,
                            "Sensors ready! You can start tapping buttons.", Toast.LENGTH_LONG).show();
                }
            });
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not used
    }

    private void updateStatus(String message) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (statusText != null) {
                    statusText.setText(message);
                }
            }
        });
    }

    private void updatePinDots() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                for (int i = 0; i < pinDots.getChildCount(); i++) {
                    View dot = pinDots.getChildAt(i);
                    GradientDrawable dotDrawable = (GradientDrawable) dot.getBackground();

                    if (i < enteredPin.size()) {
                        dotDrawable.setColor(0xFFFFFFFF);
                    } else {
                        dotDrawable.setColor(0x60FFFFFF);
                    }
                }
            }
        });
    }

    private void clearPin() {
        enteredPin.clear();
        updatePinDots();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }

        if (tcpClient != null) {
            tcpClient.disconnect();
        }

        Log.d("SensorClient", "App destroyed - Total samples received: " + totalSamplesReceived);
    }
}