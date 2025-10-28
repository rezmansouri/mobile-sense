package com.example.mobilesense;

import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import androidx.annotation.NonNull;

public class ServerConfigDialog extends Dialog {
    private EditText ipEditText, portEditText;
    private Button connectButton;
    private ServerConfigListener listener;

    public ServerConfigDialog(@NonNull Context context, ServerConfigListener listener) {
        super(context);
        this.listener = listener;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.dialog_server_config);

        setTitle("Server Configuration");
        setCancelable(false);

        ipEditText = findViewById(R.id.ipEditText);
        portEditText = findViewById(R.id.portEditText);
        connectButton = findViewById(R.id.connectButton);

        // Set default values
        ipEditText.setText("192.168.1.100");
        portEditText.setText("8080");

        connectButton.setOnClickListener(v -> {
            String ip = ipEditText.getText().toString().trim();
            String portStr = portEditText.getText().toString().trim();

            if (ip.isEmpty() || portStr.isEmpty()) {
                return;
            }

            try {
                int port = Integer.parseInt(portStr);
                if (listener != null) {
                    listener.onServerConfig(ip, port);
                }
                dismiss();
            } catch (NumberFormatException e) {
                portEditText.setError("Invalid port number");
            }
        });
    }

    public interface ServerConfigListener {
        void onServerConfig(String ip, int port);
    }
}