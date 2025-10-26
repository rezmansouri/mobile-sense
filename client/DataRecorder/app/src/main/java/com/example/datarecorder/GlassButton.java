package com.example.datarecorder;

import android.content.Context;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.widget.Button;
import androidx.appcompat.widget.AppCompatButton;

public class GlassButton extends AppCompatButton {
    private int buttonNumber;

    public GlassButton(Context context) {
        super(context);
        init();
    }

    public GlassButton(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public GlassButton(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        setBackgroundResource(R.drawable.glass_button_ripple);
        setTextColor(0xFFFFFFFF);
        setAllCaps(false);
    }

    public void setButtonNumber(int number) {
        this.buttonNumber = number;
    }

    public int getButtonNumber() {
        return buttonNumber;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            animate().scaleX(0.9f).scaleY(0.9f).setDuration(100).start();
        } else if (event.getAction() == MotionEvent.ACTION_UP ||
                event.getAction() == MotionEvent.ACTION_CANCEL) {
            animate().scaleX(1f).scaleY(1f).setDuration(100).start();
        }
        return super.onTouchEvent(event);
    }
}