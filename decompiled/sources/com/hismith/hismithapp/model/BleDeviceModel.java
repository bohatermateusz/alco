package com.hismith.hismithapp.model;

import android.bluetooth.BluetoothDevice;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class BleDeviceModel implements Cloneable {
    private String appVersion;
    private int bleType;
    private BluetoothDevice bluetoothDevice;
    private String bootVersion;
    private String deviceAddress;
    private String deviceCode;
    private String deviceName;
    private int mode;
    private int music;
    private int piu;
    private int position;
    private ProductModel productModel;
    private int reversal;
    private int runMode;
    private int smoothness;
    private int speed;
    private String stackVersion;
    private int startPosition;
    private int vibration;
    private int vibrationMode;
    private int writePosition;
    private int stopPosition = 100;
    private int battery = -1;

    public BleDeviceModel() {
    }

    public BleDeviceModel(String str, String str2, String str3, BluetoothDevice bluetoothDevice) {
        this.deviceName = str;
        this.deviceAddress = str2;
        this.deviceCode = str3;
        this.bluetoothDevice = bluetoothDevice;
    }

    public BleDeviceModel(String str, String str2, String str3, BluetoothDevice bluetoothDevice, ProductModel productModel) {
        this.deviceName = str;
        this.deviceAddress = str2;
        this.deviceCode = str3;
        this.bluetoothDevice = bluetoothDevice;
        this.productModel = productModel;
    }

    public void clearData() {
        LogUtil.e("重载模式 clearData");
        this.speed = 0;
        this.mode = 0;
        this.vibration = 0;
        this.vibrationMode = 0;
        this.music = 0;
        this.reversal = 0;
        this.piu = 0;
        this.runMode = 0;
        this.position = 0;
        this.writePosition = 0;
    }

    /* renamed from: clone, reason: merged with bridge method [inline-methods] */
    public BleDeviceModel m613clone() {
        try {
            return (BleDeviceModel) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
            return null;
        }
    }

    public String getAppVersion() {
        return this.appVersion;
    }

    public int getBattery() {
        return this.battery;
    }

    public BluetoothDevice getBluetoothDevice() {
        return this.bluetoothDevice;
    }

    public String getBootVersion() {
        return this.bootVersion;
    }

    public String getDeviceAddress() {
        return this.deviceAddress;
    }

    public String getDeviceCode() {
        return this.deviceCode;
    }

    public String getDeviceName() {
        return this.deviceName;
    }

    public int getMode() {
        return this.mode;
    }

    public int getMusic() {
        return this.music;
    }

    public int getPiu() {
        return this.piu;
    }

    public int getPosition() {
        return this.position;
    }

    public ProductModel getProductModel() {
        return this.productModel;
    }

    public int getReversal() {
        return this.reversal;
    }

    public int getRunMode() {
        return this.runMode;
    }

    public int getSmoothness() {
        return this.smoothness;
    }

    public int getSpeed() {
        return this.speed;
    }

    public String getStackVersion() {
        return this.stackVersion;
    }

    public int getStartPosition() {
        return this.startPosition;
    }

    public int getStopPosition() {
        return this.stopPosition;
    }

    public int getVibration() {
        return this.vibration;
    }

    public int getVibrationMode() {
        return this.vibrationMode;
    }

    public int getWritePosition() {
        return this.writePosition;
    }

    public void setAppVersion(String str) {
        this.appVersion = str;
    }

    public void setBattery(int i) {
        this.battery = i;
    }

    public void setBluetoothDevice(BluetoothDevice bluetoothDevice) {
        this.bluetoothDevice = bluetoothDevice;
    }

    public void setBootVersion(String str) {
        this.bootVersion = str;
    }

    public void setDeviceAddress(String str) {
        this.deviceAddress = str;
    }

    public void setDeviceCode(String str) {
        this.deviceCode = str;
    }

    public void setDeviceName(String str) {
        this.deviceName = str;
    }

    public void setMode(int i) {
        this.mode = i;
    }

    public void setMusic(int i) {
        this.music = i;
    }

    public void setPiu(int i) {
        this.piu = i;
    }

    public void setPosition(int i) {
        this.position = i;
    }

    public void setProductModel(ProductModel productModel) {
        this.productModel = productModel;
    }

    public void setReversal(int i) {
        this.reversal = i;
    }

    public void setRunMode(int i) {
        this.runMode = i;
    }

    public void setSmoothness(int i) {
        this.smoothness = i;
    }

    public void setSpeed(int i) {
        this.speed = i;
    }

    public void setStackVersion(String str) {
        this.stackVersion = str;
    }

    public void setStartPosition(int i) {
        this.startPosition = i;
    }

    public void setStopPosition(int i) {
        this.stopPosition = i;
    }

    public void setVibration(int i) {
        this.vibration = i;
    }

    public void setVibrationMode(int i) {
        this.vibrationMode = i;
    }

    public void setWritePosition(int i) {
        this.writePosition = i;
    }

    public String toString() {
        return "BleDeviceModel{bleType=" + this.bleType + ", speed=" + this.speed + ", mode=" + this.mode + ", vibration=" + this.vibration + ", vibrationMode=" + this.vibrationMode + ", music=" + this.music + ", reversal=" + this.reversal + ", piu=" + this.piu + ", runMode=" + this.runMode + ", writePosition=" + this.writePosition + ", position=" + this.position + ", stopPosition=" + this.stopPosition + ", startPosition=" + this.startPosition + ", smoothness=" + this.smoothness + ", battery=" + this.battery + ", bootVersion='" + this.bootVersion + "', stackVersion='" + this.stackVersion + "', appVersion='" + this.appVersion + "', deviceCode='" + this.deviceCode + "', deviceName='" + this.deviceName + "', deviceAddress='" + this.deviceAddress + "', bluetoothDevice=" + this.bluetoothDevice + ", productModel=" + this.productModel + '}';
    }
}
