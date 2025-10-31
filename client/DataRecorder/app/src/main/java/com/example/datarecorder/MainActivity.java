package com.example.datarecorder;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.drawable.GradientDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private LinearLayout pinDots;
    private List<Integer> enteredPin = new ArrayList<>();
    private static final int MAX_PIN_LENGTH = 4;

    // Sensor related variables
    private SensorManager sensorManager;
    private Sensor accelerometer, gyroscope, rotationVector;
    private static final int SAMPLING_RATE = 100;
    private static final int WINDOW_SIZE = 81;
    private static final int BUFFER_SIZE = 2000;

    // Circular buffer for sensor data
    private float[][] sensorBuffer = new float[9][BUFFER_SIZE];
    private long[] timestampBuffer = new long[BUFFER_SIZE];
    private int bufferIndex = 0;
    private boolean isBufferFilled = false;
    private int totalSamplesReceived = 0;

    // CSV file management
    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024;
    private static final int MAX_RECORDS_PER_FILE = 5000;
    private int currentRecordCount = 0;
    private File csvFile;
    private FileWriter fileWriter;
    private SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault());
    private int fileCounter = 1;
    private boolean hasStoragePermission = false;

    // Permissions
    private static final int STORAGE_PERMISSION_CODE = 100;

    // UI Elements
    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeViews();
        createPinDots();
        setupButtonClicks();

        // Start sensors immediately
        setupSensors();

        // Check storage permissions
        checkStoragePermissions();

        Log.d("SensorData", "=== APP STARTED ===");
    }

    private void initializeViews() {
        pinDots = findViewById(R.id.pinDots);
        statusText = findViewById(R.id.statusText);
    }

    private void checkStoragePermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            hasStoragePermission = true;
            createNewCSVFile();
            updateStatus("Ready - Tap buttons to record data");
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED) {
                hasStoragePermission = true;
                createNewCSVFile();
                updateStatus("Ready - Tap buttons to record data");
            } else {
                ActivityCompat.requestPermissions(this,
                        new String[]{
                                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                                Manifest.permission.READ_EXTERNAL_STORAGE
                        },
                        STORAGE_PERMISSION_CODE);
                updateStatus("Requesting storage permission...");
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == STORAGE_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                hasStoragePermission = true;
                createNewCSVFile();
                updateStatus("Ready - Tap buttons to record data");
                Toast.makeText(this, "Storage permission granted", Toast.LENGTH_SHORT).show();
            } else {
                hasStoragePermission = false;
                createNewCSVFile();
                updateStatus("Ready - Using app storage");
                Toast.makeText(this, "Using app-specific storage", Toast.LENGTH_LONG).show();
            }
        }
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

    private void setupSensors() {
        try {
            sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

            if (sensorManager != null) {
                Log.d("SensorData", "=== SENSOR MANAGER INITIALIZED ===");

                // Get ALL available sensors for comprehensive logging
                List<Sensor> allSensors = sensorManager.getSensorList(Sensor.TYPE_ALL);
                Log.d("SensorData", "=== ALL AVAILABLE SENSORS ON THIS DEVICE ===");
                Log.d("SensorData", "Total sensors found: " + allSensors.size());

                for (Sensor sensor : allSensors) {
                    Log.d("SensorData", String.format("Sensor: %-25s | Type: %-30s | Vendor: %-15s | Version: %d | Power: %.1fmA | MaxRange: %.3f",
                            sensor.getName(),
                            getSensorTypeName(sensor.getType()),
                            sensor.getVendor(),
                            sensor.getVersion(),
                            sensor.getPower(),
                            sensor.getMaximumRange()));
                }

                // Get our specific required sensors
                Log.d("SensorData", "=== TARGET SENSORS STATUS ===");

                accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
                gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
                rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

                logSensorStatus("ACCELEROMETER", accelerometer, Sensor.TYPE_ACCELEROMETER);
                logSensorStatus("GYROSCOPE", gyroscope, Sensor.TYPE_GYROSCOPE);
                logSensorStatus("ROTATION VECTOR", rotationVector, Sensor.TYPE_ROTATION_VECTOR);

                // Register listeners and verify registration
                Log.d("SensorData", "=== SENSOR REGISTRATION STATUS ===");
                boolean anySensorRegistered = false;

                if (accelerometer != null) {
                    boolean registered = sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
                    Log.d("SensorData", "Accelerometer registration: " + (registered ? "SUCCESS" : "FAILED"));
                    if (registered) anySensorRegistered = true;
                }

                if (gyroscope != null) {
                    boolean registered = sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME);
                    Log.d("SensorData", "Gyroscope registration: " + (registered ? "SUCCESS" : "FAILED"));
                    if (registered) anySensorRegistered = true;
                }

                if (rotationVector != null) {
                    boolean registered = sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_GAME);
                    Log.d("SensorData", "Rotation Vector registration: " + (registered ? "SUCCESS" : "FAILED"));
                    if (registered) anySensorRegistered = true;
                }

                Log.d("SensorData", "Overall sensor registration: " + (anySensorRegistered ? "SUCCESS" : "FAILED - NO SENSORS WORKING"));

                if (!anySensorRegistered) {
                    Log.e("SensorData", "CRITICAL: NO SENSORS COULD BE REGISTERED!");
                    updateStatus("Error: No sensors available");

                    // Try to register any available motion sensor as fallback
                    List<Sensor> motionSensors = sensorManager.getSensorList(Sensor.TYPE_ALL);
                    for (Sensor sensor : motionSensors) {
                        if (sensor.getName().toLowerCase().contains("accel") ||
                                sensor.getName().toLowerCase().contains("gyro") ||
                                sensor.getName().toLowerCase().contains("rotation")) {
                            Log.d("SensorData", "Trying fallback registration for: " + sensor.getName());
                            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL);
                        }
                    }
                } else {
                    updateStatus("Sensors initialized - Wait for data...");
                    Log.d("SensorData", "=== SENSORS READY - WAITING FOR DATA ===");
                }

            } else {
                Log.e("SensorData", "CRITICAL: SensorManager is null - Sensors not available on this device");
                updateStatus("Error: Sensor service unavailable");
            }
        } catch (Exception e) {
            Log.e("SensorData", "EXCEPTION in sensor setup: " + e.getMessage(), e);
            updateStatus("Sensor error: " + e.getMessage());
        }
    }

    private void logSensorStatus(String sensorName, Sensor sensor, int sensorType) {
        if (sensor != null) {
            Log.d("SensorData", String.format("%-18s: FOUND - %s (Range: ±%.3f, Power: %.1fmA)",
                    sensorName, sensor.getName(), sensor.getMaximumRange(), sensor.getPower()));
        } else {
            Log.e("SensorData", String.format("%-18s: NOT FOUND - This sensor is not available on your device", sensorName));
        }
    }

    private String getSensorTypeName(int sensorType) {
        switch (sensorType) {
            case Sensor.TYPE_ACCELEROMETER: return "ACCELEROMETER";
            case Sensor.TYPE_GYROSCOPE: return "GYROSCOPE";
            case Sensor.TYPE_ROTATION_VECTOR: return "ROTATION_VECTOR";
            case Sensor.TYPE_MAGNETIC_FIELD: return "MAGNETIC_FIELD";
            case Sensor.TYPE_LIGHT: return "LIGHT";
            case Sensor.TYPE_PROXIMITY: return "PROXIMITY";
            case Sensor.TYPE_GRAVITY: return "GRAVITY";
            case Sensor.TYPE_LINEAR_ACCELERATION: return "LINEAR_ACCELERATION";
            case Sensor.TYPE_STEP_COUNTER: return "STEP_COUNTER";
            case Sensor.TYPE_GAME_ROTATION_VECTOR: return "GAME_ROTATION_VECTOR";
            case Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR: return "GEOMAGNETIC_ROTATION_VECTOR";
            default: return "UNKNOWN(" + sensorType + ")";
        }
    }

    private void createNewCSVFile() {
        closeCurrentFile();

        try {
            File directory = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
            if (directory == null) {
                directory = getFilesDir();
            }

            if (directory != null) {
                File sensorDataDir = new File(directory, "SensorData");
                if (!sensorDataDir.exists()) {
                    sensorDataDir.mkdirs();
                }

                String baseName = "tap_data_" + dateFormat.format(new Date());
                String fileName = String.format(Locale.getDefault(), "%s_%03d.csv", baseName, fileCounter);
                csvFile = new File(sensorDataDir, fileName);
                fileWriter = new FileWriter(csvFile, true);
                currentRecordCount = 0;

                // Write CSV header
                String header = "digit,timestamp,file_index";
                for (int sensor = 0; sensor < 9; sensor++) {
                    for (int sample = 0; sample < WINDOW_SIZE; sample++) {
                        header += String.format(",sensor_%d_sample_%d", sensor, sample);
                    }
                }
                fileWriter.write(header + "\n");
                fileWriter.flush();

                Log.d("SensorData", "Created new CSV file: " + csvFile.getAbsolutePath());
                Log.d("SensorData", "File path: " + csvFile.getAbsolutePath());

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this,
                                "New file created: " + fileName, Toast.LENGTH_SHORT).show();
                    }
                });

                updateStatus("Ready - File: " + fileName);
            }
        } catch (IOException e) {
            Log.e("SensorData", "Error creating CSV file", e);
            updateStatus("Error creating file");
        }
    }

    private void closeCurrentFile() {
        if (fileWriter != null) {
            try {
                fileWriter.close();
                Log.d("SensorData", "Closed current CSV file");
            } catch (IOException e) {
                Log.e("SensorData", "Error closing CSV file", e);
            }
            fileWriter = null;
        }
    }

    private boolean shouldCreateNewFile() {
        if (fileWriter == null) return true;
        if (currentRecordCount >= MAX_RECORDS_PER_FILE) return true;
        if (csvFile != null && csvFile.exists() && csvFile.length() >= MAX_FILE_SIZE) return true;
        return false;
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
                        Log.d("SensorData", "Button pressed: " + digit);
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

            Log.d("SensorData", "Capturing data for digit: " + number +
                    ", Buffer filled: " + isBufferFilled +
                    ", Total samples: " + totalSamplesReceived);

            // Capture sensor data for this tap
            captureSensorData(number);

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

    private void captureSensorData(int digit) {
        if (!isBufferFilled) {
            Log.w("SensorData", "Buffer not filled yet! Only have " + totalSamplesReceived + " samples");
            Toast.makeText(this, "Waiting for sensor data...", Toast.LENGTH_SHORT).show();
            return;
        }

        if (shouldCreateNewFile()) {
            fileCounter++;
            createNewCSVFile();
        }

        float[][] tapData = extractSensorWindow();

        if (tapData != null) {
            saveToCSV(digit, tapData);
        } else {
            Log.e("SensorData", "Failed to extract sensor window");
        }
    }

    private float[][] extractSensorWindow() {
        int startIndex = (bufferIndex - WINDOW_SIZE + BUFFER_SIZE) % BUFFER_SIZE;
        float[][] window = new float[9][WINDOW_SIZE];

        Log.d("SensorData", "Extracting window - Start: " + startIndex +
                ", Current: " + bufferIndex + ", Size: " + WINDOW_SIZE);

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
            Log.w("SensorData", "Extracted window contains only zeros!");
            return null;
        }

        return window;
    }

    private void saveToCSV(int digit, float[][] sensorData) {
        if (fileWriter == null) {
            Log.e("SensorData", "File writer is null, cannot save!");
            return;
        }

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    long timestamp = System.currentTimeMillis();
                    StringBuilder csvLine = new StringBuilder();
                    csvLine.append(digit).append(",")
                            .append(timestamp).append(",")
                            .append(fileCounter);

                    for (int sensor = 0; sensor < 9; sensor++) {
                        for (int sample = 0; sample < WINDOW_SIZE; sample++) {
                            csvLine.append(",").append(sensorData[sensor][sample]);
                        }
                    }

                    fileWriter.write(csvLine.toString() + "\n");
                    fileWriter.flush();

                    currentRecordCount++;

                    final String logMessage = "SUCCESS: Saved tap #" + currentRecordCount +
                            " for digit: " + digit +
                            " to file: " + fileCounter +
                            " - File: " + csvFile.getName();

                    Log.d("SensorData", logMessage);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            updateStatus("Saved: " + digit + " (Total: " + currentRecordCount + ")");
                        }
                    });

                } catch (IOException e) {
                    Log.e("SensorData", "ERROR writing to CSV: " + e.getMessage(), e);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this,
                                    "Error saving data", Toast.LENGTH_SHORT).show();
                        }
                    });
                }
            }
        }).start();
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        totalSamplesReceived++;

        // Log detailed information about the first 10 samples from each sensor
        if (totalSamplesReceived <= 10) {
            Log.d("SensorData", String.format("SAMPLE #%d - Sensor: %-25s | Type: %s | Values: [%.3f, %.3f, %.3f] | Accuracy: %d",
                    totalSamplesReceived,
                    event.sensor.getName(),
                    getSensorTypeName(event.sensor.getType()),
                    event.values[0], event.values[1], event.values[2],
                    event.accuracy));
        }

        int sensorType = event.sensor.getType();
        int baseIndex = -1;

        if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            baseIndex = 0;
        } else if (sensorType == Sensor.TYPE_GYROSCOPE) {
            baseIndex = 3;
        } else if (sensorType == Sensor.TYPE_ROTATION_VECTOR) {
            baseIndex = 6;
        } else {
            // Log other sensor types that we might be receiving
            if (totalSamplesReceived <= 5) {
                Log.d("SensorData", "Received data from additional sensor: " +
                        event.sensor.getName() + " Type: " + getSensorTypeName(sensorType));
            }
            return;
        }

        // Store sensor data
        for (int i = 0; i < 3 && i < event.values.length; i++) {
            sensorBuffer[baseIndex + i][bufferIndex] = event.values[i];
        }

        timestampBuffer[bufferIndex] = System.currentTimeMillis();
        bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;

        // Set buffer as filled after we have enough samples
        if (totalSamplesReceived >= WINDOW_SIZE && !isBufferFilled) {
            isBufferFilled = true;
            Log.d("SensorData", "*** BUFFER NOW FILLED! ***");
            Log.d("SensorData", "Total samples received: " + totalSamplesReceived);
            Log.d("SensorData", "Sensors are actively providing data");
            Log.d("SensorData", "Ready to capture tap events");

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    updateStatus("Ready - Sensors active (" + totalSamplesReceived + " samples)");
                    Toast.makeText(MainActivity.this,
                            "Sensors ready! You can start tapping buttons.", Toast.LENGTH_LONG).show();
                }
            });
        }

        // Log every 100 samples to show we're receiving data
        if (totalSamplesReceived % 100 == 0) {
            Log.d("SensorData", "Received " + totalSamplesReceived + " sensor samples so far");
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        Log.d("SensorData", "Accuracy changed for " + sensor.getName() + ": " + accuracy);
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
            Log.d("SensorData", "Unregistered all sensor listeners");
        }

        closeCurrentFile();

        Log.d("SensorData", "=== APP DESTROYED ===");
        Log.d("SensorData", "Final statistics - Total samples received: " + totalSamplesReceived);
        Log.d("SensorData", "Final statistics - Total records saved: " + currentRecordCount);
    }
}