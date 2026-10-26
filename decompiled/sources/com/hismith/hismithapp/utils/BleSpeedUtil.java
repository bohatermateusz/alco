package com.hismith.hismithapp.utils;

import android.content.Context;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.model.BleDeviceModel;

/* loaded from: classes3.dex */
public class BleSpeedUtil {
    private String deviceCode;
    private long lastTime;
    private float lastX;
    private float lastY;
    private float lastZ;
    private Context mContext;
    private boolean isShake = false;
    private long timeInterval = 100;
    private boolean isInitPosition = false;

    public interface OnShakeListener {
        void onShake(int i);
    }

    public BleSpeedUtil(Context context) {
        this.mContext = context;
    }

    public boolean isShake() {
        return this.isShake;
    }

    public void setShake(boolean z) {
        this.isShake = z;
    }

    public void shakeSpeed(float f, float f2, float f3) {
        shakeSpeed(f, f2, f3, null);
    }

    public void shakeSpeed(float f, float f2, float f3, int i, OnShakeListener onShakeListener) {
        if (this.isShake) {
            if (this.timeInterval > System.currentTimeMillis() - this.lastTime) {
                return;
            }
            if (!this.isInitPosition) {
                this.lastX = f;
                this.lastY = f2;
                this.lastZ = f3;
                this.isInitPosition = true;
            }
            float f4 = f - this.lastX;
            float f5 = f2 - this.lastY;
            float f6 = f3 - this.lastZ;
            this.lastX = f;
            this.lastY = f2;
            this.lastZ = f3;
            int sqrt = (int) (((((Math.sqrt(((f4 * f4) + (f5 * f5)) + (f6 * f6)) / this.timeInterval) * 1000.0d) - 10.0d) / 490.0d) * i);
            if (sqrt <= i) {
                i = sqrt < 0 ? 0 : sqrt;
            }
            BleDeviceModel bleDeviceModel = BleManager.getInstance(this.mContext).getmConnBleDevice();
            if (onShakeListener != null) {
                onShakeListener.onShake(i);
            } else if (bleDeviceModel == null || i != bleDeviceModel.getSpeed()) {
                BleManager.getInstance(this.mContext).writeSpeed(i);
            }
            this.lastTime = System.currentTimeMillis();
        }
    }

    public void shakeSpeed(float f, float f2, float f3, OnShakeListener onShakeListener) {
        shakeSpeed(f, f2, f3, MyApplication.getApplication().getmProductModel() != null ? MyApplication.getApplication().getmProductModel().getShake() : 100, onShakeListener);
    }
}
