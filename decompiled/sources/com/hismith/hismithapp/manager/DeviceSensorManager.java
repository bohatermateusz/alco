package com.hismith.hismithapp.manager;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

/* loaded from: classes3.dex */
public class DeviceSensorManager {
    private static DeviceSensorManager instance;
    private Context mContext;
    private DeviceSensorEventListener mListener;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private boolean isReversal = false;
    private long triggerTime = 0;
    private int shakeCount = 0;
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;

    class DeviceSensorEventListener implements SensorEventListener {
        DeviceSensorEventListener() {
        }

        @Override // android.hardware.SensorEventListener
        public void onAccuracyChanged(Sensor sensor, int i) {
        }

        @Override // android.hardware.SensorEventListener
        public void onSensorChanged(SensorEvent sensorEvent) {
        }
    }

    public DeviceSensorManager(Context context) {
        this.mContext = context;
    }

    public static DeviceSensorManager getInstance(Context context) {
        DeviceSensorManager deviceSensorManager;
        synchronized (DeviceSensorManager.class) {
            deviceSensorManager = instance;
            if (deviceSensorManager == null) {
                deviceSensorManager = new DeviceSensorManager(context);
                instance = deviceSensorManager;
            }
        }
        return deviceSensorManager;
    }

    public boolean isReversal() {
        return this.isReversal;
    }

    public void register() {
        Sensor sensor = this.mSensor;
        if (sensor != null) {
            this.mSensorManager.registerListener(this.mListener, sensor, 2);
        }
    }

    public void release() {
        if (this.mSensor != null) {
            this.mSensorManager.unregisterListener(this.mListener);
        }
    }
}
