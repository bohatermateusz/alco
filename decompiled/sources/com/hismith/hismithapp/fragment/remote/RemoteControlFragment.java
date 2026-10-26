package com.hismith.hismithapp.fragment.remote;

import android.graphics.Point;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.LazyLoadFragment;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.utils.BleSpeedUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public abstract class RemoteControlFragment extends LazyLoadFragment implements SensorEventListener {
    public BaseActivity baseActivity;
    private BleSpeedUtil mBleSpeedUtil;
    public List<ProductModel.Mode> mModes;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    public List<ProductModel.Mode> mVibrationModes;
    private RecorderVolumeManager recorderVolumeManager;
    private BleDeviceModel recoverBleDevice;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.remote.RemoteControlFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                int mode = RemoteControlFragment.this.recoverBleDevice.getMode();
                RemoteControlFragment remoteControlFragment = RemoteControlFragment.this;
                if (mode <= 0) {
                    if (remoteControlFragment.recoverBleDevice.getSpeed() <= 0) {
                        return false;
                    }
                    BleManager.getInstance(RemoteControlFragment.this.baseActivity).writeSpeed(RemoteControlFragment.this.recoverBleDevice.getSpeed());
                    return false;
                }
                int size = remoteControlFragment.mModes.size();
                int mode2 = RemoteControlFragment.this.recoverBleDevice.getMode();
                RemoteControlFragment remoteControlFragment2 = RemoteControlFragment.this;
                if (size >= mode2) {
                    BleManager.getInstance(remoteControlFragment2.baseActivity).writeData(RemoteControlFragment.this.mModes.get(RemoteControlFragment.this.recoverBleDevice.getMode() - 1).getmCommand());
                    return false;
                }
                BleManager.getInstance(remoteControlFragment2.baseActivity).writeSpeedMode(RemoteControlFragment.this.recoverBleDevice.getMode());
                return false;
            }
            if (i != 2) {
                if (i == 3) {
                    BleManager.getInstance(RemoteControlFragment.this.baseActivity).writeOpenVoice();
                    return false;
                }
                if (i == 4) {
                    BleManager.getInstance(RemoteControlFragment.this.baseActivity).writeRunManual();
                    return false;
                }
                if (i != 5) {
                    return false;
                }
                BleManager.getInstance(RemoteControlFragment.this.baseActivity).writePosition(RemoteControlFragment.this.recoverBleDevice.getWritePosition());
                return false;
            }
            int vibrationMode = RemoteControlFragment.this.recoverBleDevice.getVibrationMode();
            RemoteControlFragment remoteControlFragment3 = RemoteControlFragment.this;
            if (vibrationMode <= 0) {
                if (remoteControlFragment3.recoverBleDevice.getVibration() <= 0) {
                    return false;
                }
                BleManager.getInstance(RemoteControlFragment.this.baseActivity).writeVibrate(RemoteControlFragment.this.recoverBleDevice.getVibration());
                return false;
            }
            int size2 = remoteControlFragment3.mVibrationModes.size();
            int vibrationMode2 = RemoteControlFragment.this.recoverBleDevice.getVibrationMode();
            RemoteControlFragment remoteControlFragment4 = RemoteControlFragment.this;
            if (size2 >= vibrationMode2) {
                BleManager.getInstance(remoteControlFragment4.baseActivity).writeData(RemoteControlFragment.this.mVibrationModes.get(RemoteControlFragment.this.recoverBleDevice.getVibrationMode() - 1).getmCommand());
                return false;
            }
            BleManager.getInstance(remoteControlFragment4.baseActivity).writeVibrateMode(RemoteControlFragment.this.recoverBleDevice.getVibrationMode());
            return false;
        }
    });
    private Handler handler = new Handler();
    private int delayMillis = 0;
    private int defDelayMillils = 100;
    private boolean isReversal = false;
    private long triggerTime = 0;
    private int shakeCount = 0;
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;

    private void initBleData() {
        BleDeviceModel bleDeviceModel = BleManager.getInstance(this.baseActivity).getmConnBleDevice();
        if (bleDeviceModel != null) {
            refreshSpeedUI(bleDeviceModel.getSpeed());
            refreshSpeedModeUI(bleDeviceModel.getMode());
            refreshVibrateUI(bleDeviceModel.getVibration());
            refreshVibrateModeUI(bleDeviceModel.getVibrationMode());
            refreshMusicUI(bleDeviceModel.getMusic());
            refreshAutoUI(bleDeviceModel.getRunMode());
            refreshSmoothnessUI(bleDeviceModel.getSmoothness());
            refreshPositionUI(bleDeviceModel.getPosition());
            refreshReversalRotation(bleDeviceModel.getReversal());
        }
    }

    private void initSensor() {
        SensorManager sensorManager = (SensorManager) this.baseActivity.getSystemService("sensor");
        this.mSensorManager = sensorManager;
        Sensor defaultSensor = sensorManager.getDefaultSensor(1);
        this.mSensor = defaultSensor;
        this.mSensorManager.registerListener(this, defaultSensor, 2);
        this.mBleSpeedUtil = new BleSpeedUtil(this.baseActivity);
    }

    private void recoverBle() {
        if (this.recoverBleDevice != null) {
            this.mHandler.sendEmptyMessageDelayed(1, this.delayMillis);
            int i = this.delayMillis + this.defDelayMillils;
            this.delayMillis = i;
            this.mHandler.sendEmptyMessageDelayed(2, i);
            this.delayMillis += this.defDelayMillils;
            if (this.recoverBleDevice.getMusic() == 1) {
                this.mHandler.sendEmptyMessageDelayed(3, this.delayMillis);
                this.delayMillis += this.defDelayMillils;
            }
            if (this.recoverBleDevice.getRunMode() == 1) {
                this.mHandler.sendEmptyMessageDelayed(4, this.delayMillis);
                this.delayMillis += this.defDelayMillils;
            }
            if (this.recoverBleDevice.getWritePosition() > 0) {
                this.mHandler.sendEmptyMessageDelayed(5, this.delayMillis);
            }
            this.delayMillis = 0;
        }
    }

    public void init() {
        initSensor();
        initBleData();
        this.recorderVolumeManager = new RecorderVolumeManager();
    }

    public boolean isShake() {
        BleSpeedUtil bleSpeedUtil = this.mBleSpeedUtil;
        if (bleSpeedUtil != null) {
            return bleSpeedUtil.isShake();
        }
        return false;
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // android.hardware.SensorEventListener
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        if (this.mSensor != null) {
            this.mSensorManager.unregisterListener(this);
        }
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.cancelRecording();
        }
        ToastUtil.cancelToast();
    }

    @Override // android.hardware.SensorEventListener
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == 1) {
            float f = sensorEvent.values[0];
            float f2 = sensorEvent.values[1];
            float f3 = sensorEvent.values[2];
            long currentTimeMillis = System.currentTimeMillis() / 1000;
            this.mBleSpeedUtil.shakeSpeed(f, f2, f3);
            if (f3 <= -9.0f && f3 > -10.0f) {
                int i = this.reversalCount + 1;
                this.reversalCount = i;
                if (i >= 4) {
                    this.recoverCount = 0;
                    if (!this.isReversal) {
                        LogUtil.e("Sensor value 翻转翻转翻转");
                        this.isReversal = true;
                        this.recoverBleDevice = BleManager.getInstance(this.baseActivity).getmConnBleDevice().m613clone();
                        BleManager.getInstance(this.baseActivity).writeShutdown();
                    }
                }
            } else if (f3 > -1.0f) {
                int i2 = this.recoverCount + 1;
                this.recoverCount = i2;
                if (i2 >= 4) {
                    this.reversalCount = 0;
                    if (this.isReversal) {
                        LogUtil.e("Sensor value 翻转恢复");
                        this.isReversal = false;
                        recoverBle();
                    }
                }
            }
            if (this.reversalTime != currentTimeMillis) {
                this.reversalTime = currentTimeMillis;
                this.reversalCount = 0;
                this.recoverCount = 0;
            }
        }
    }

    public abstract void refreshAutoUI(int i);

    public abstract void refreshLinePoints(Map<String, Point> map);

    public abstract void refreshMusicUI(int i);

    public abstract void refreshPositionUI(int i);

    public void refreshReversalRotation(int i) {
    }

    public abstract void refreshSmoothnessUI(int i);

    public abstract void refreshSpeedModeUI(int i);

    public abstract void refreshSpeedUI(int i);

    public abstract void refreshStartPositionUI(int i);

    public abstract void refreshStopPositionUi(int i);

    public abstract void refreshVibrateLinePoints(Map<String, Point> map);

    public abstract void refreshVibrateModeUI(int i);

    public abstract void refreshVibrateUI(int i);

    public abstract void refreshWebSpeedUI(int i);

    public abstract void refreshWebVibrateUI(int i);

    public void setListener() {
        Sensor sensor = this.mSensor;
        if (sensor != null) {
            this.mSensorManager.registerListener(this, sensor, 2);
        }
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteControlFragment.2
            @Override // com.hismith.hismithapp.manager.RecorderVolumeManager.OnMicStatusListener
            public void onMicAmplitude(double d) {
                double d2 = d - 55.0d;
                if (d2 < 0.0d) {
                    d2 = 0.0d;
                }
                int i = (int) (d2 * 2.0d);
                if (i > 100) {
                    i = 100;
                }
                BleDeviceModel bleDeviceModel = BleManager.getInstance(RemoteControlFragment.this.baseActivity).getmConnBleDevice();
                if (bleDeviceModel == null || i != bleDeviceModel.getSpeed()) {
                    BleManager.getInstance(RemoteControlFragment.this.baseActivity).writeSpeed(i);
                }
            }
        });
    }

    public abstract void startHeart();

    public void startMicRecording() {
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.startRecording();
        }
    }

    public void startShake() {
        BleSpeedUtil bleSpeedUtil = this.mBleSpeedUtil;
        if (bleSpeedUtil != null) {
            bleSpeedUtil.setShake(true);
        }
    }

    public abstract void stopHeart();

    public void stopMicRecording() {
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.stopRecording();
        }
        BleManager.getInstance(this.baseActivity).writeSpeed(0);
    }

    public void stopShake() {
        BleSpeedUtil bleSpeedUtil = this.mBleSpeedUtil;
        if (bleSpeedUtil != null) {
            bleSpeedUtil.setShake(false);
        }
    }
}
