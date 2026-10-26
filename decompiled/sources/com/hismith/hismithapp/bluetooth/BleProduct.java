package com.hismith.hismithapp.bluetooth;

import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import com.facebook.appevents.integrity.IntegrityManager;
import com.google.gson.Gson;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitApi;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.manager.ProductManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.ProductsModel;
import com.hismith.hismithapp.product.Factory;
import com.hismith.hismithapp.product.ProductAAControl;
import com.hismith.hismithapp.product.ProductBaseControl;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HexUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ModeDataVCUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StringUtils;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

/* loaded from: classes3.dex */
public class BleProduct {
    public static final String BLE_STATE = "ble_state";
    private BluetoothDevice mBluetoothDevice;
    private BluetoothGatt mBluetoothGatt;
    private BleDeviceModel mConnBleDevice;
    private ConnectionStateCallback mConnectionStateCallback;
    private Context mContext;
    private String mDeviceCode;
    private String mDeviceName;
    private ProductBaseControl mProductBaseControl;
    private ProductModel mProductModel;
    private int mStatus = 0;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.bluetooth.BleProduct.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            try {
                if (message.what != 9) {
                    return false;
                }
                BleProduct bleProduct = BleProduct.this;
                bleProduct.writeSmooth(bleProduct.mConnBleDevice.getSmoothness());
                return false;
            } catch (Exception e) {
                e.printStackTrace();
                return false;
            }
        }
    });
    private final BluetoothGattCallback mGattCallback = new BluetoothGattCallback() { // from class: com.hismith.hismithapp.bluetooth.BleProduct.2
        private String tempReceiver;

        private void setBleNotification() {
            BluetoothGattCharacteristic characteristic;
            BluetoothGattDescriptor descriptor;
            if (BleProduct.this.mBluetoothGatt == null || BleProduct.this.mBluetoothGatt.getService(Constans.UUID_SERIAL_DATA) == null || (characteristic = BleProduct.this.mBluetoothGatt.getService(Constans.UUID_SERIAL_DATA).getCharacteristic(Constans.UUID_SERIAL_DATA_NOTIFY)) == null) {
                return;
            }
            boolean characteristicNotification = BleProduct.this.mBluetoothGatt.setCharacteristicNotification(characteristic, true);
            LogUtil.e(BleProduct.this.mDeviceName + "蓝牙广播通知：" + characteristicNotification);
            if (!characteristicNotification || (descriptor = characteristic.getDescriptor(UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"))) == null) {
                return;
            }
            descriptor.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
            LogUtil.e(BleProduct.this.mDeviceName + "蓝牙广播通知描述写入：" + BleProduct.this.mBluetoothGatt.writeDescriptor(descriptor));
        }

        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        /* JADX WARN: Code restructure failed: missing block: B:58:0x01a2, code lost:
        
            if (r14.equals("03") != false) goto L49;
         */
        /* JADX WARN: Failed to restore switch over string. Please report as a decompilation issue */
        @Override // android.bluetooth.BluetoothGattCallback
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void onCharacteristicChanged(BluetoothGatt bluetoothGatt, BluetoothGattCharacteristic bluetoothGattCharacteristic) {
            boolean z;
            String[] strArr;
            String str;
            String str2;
            char c;
            boolean z2;
            boolean z3;
            String str3;
            super.onCharacteristicChanged(bluetoothGatt, bluetoothGattCharacteristic);
            byte[] value = bluetoothGattCharacteristic.getValue();
            String upperCase = HexUtil.bytesToHex(value).toUpperCase();
            LogUtil.e(BleProduct.this.mDeviceName + "接收:" + upperCase + ", byte.length:" + value.length + ", str.length:" + upperCase.length());
            String str4 = "CA";
            if (!TextUtils.isEmpty(this.tempReceiver)) {
                if (upperCase.startsWith("AA") || upperCase.startsWith("CC") || upperCase.startsWith("CA")) {
                    LogUtil.e(BleProduct.this.mDeviceName + "接收到标准头，清空上一次剩余字符：" + this.tempReceiver + "。不进行拼接" + upperCase);
                } else {
                    String str5 = this.tempReceiver + upperCase;
                    LogUtil.e(BleProduct.this.mDeviceName + "拼接上一次剩余字符：" + this.tempReceiver + ", 最终指令返回：" + str5);
                    upperCase = str5;
                }
                this.tempReceiver = "";
            }
            int i = 8;
            int i2 = 0;
            if (!TextUtils.isEmpty(upperCase) && upperCase.contains("AA")) {
                upperCase = upperCase.substring(upperCase.indexOf("AA"));
                String[] cut = StringUtils.cut(upperCase, 8);
                boolean z4 = false;
                int i3 = 0;
                while (i3 < cut.length) {
                    if (cut[i3].length() != i) {
                        str3 = upperCase;
                        this.tempReceiver = cut[i3];
                        LogUtil.e(BleProduct.this.mDeviceName + "多余留到下一次：" + this.tempReceiver);
                    } else {
                        if (!cut[i3].substring(i2, 2).toUpperCase().equals("AA")) {
                            return;
                        }
                        String substring = cut[i3].substring(2, 4);
                        str3 = upperCase;
                        String substring2 = cut[i3].substring(4, 6);
                        cut[i3].substring(6, 8);
                        try {
                            int intValue = Integer.valueOf(substring2, 16).intValue();
                            switch (substring.hashCode()) {
                                case 1538:
                                    if (substring.equals("02") && BleProduct.this.mConnBleDevice != null) {
                                        BleProduct.this.mConnBleDevice.clearData();
                                        break;
                                    }
                                    break;
                                case 1540:
                                    if (substring.equals("04")) {
                                        if (BleProduct.this.mConnBleDevice != null) {
                                            BleProduct.this.mConnBleDevice.setSpeed(intValue);
                                            BleProduct.this.mConnBleDevice.setMode(0);
                                        }
                                        BleProduct.this.followerControlRuleSpeed(intValue);
                                        break;
                                    }
                                    break;
                                case 1541:
                                    if (substring.equals("05") && BleProduct.this.mConnBleDevice != null) {
                                        BleProduct.this.mConnBleDevice.setMode(intValue);
                                        break;
                                    }
                                    break;
                                case 1542:
                                    if (substring.equals("06") && BleProduct.this.mConnBleDevice != null) {
                                        BleProduct.this.mConnBleDevice.setVibrationMode(intValue);
                                        break;
                                    }
                                    break;
                                case 1543:
                                    if (substring.equals("07") && BleProduct.this.mConnBleDevice != null) {
                                        BleProduct.this.mConnBleDevice.setMusic(intValue);
                                        break;
                                    }
                                    break;
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        if (BleProduct.this.mConnectionStateCallback != null) {
                            BleProduct.this.mConnectionStateCallback.onNotifyChange_v2("AA", substring, substring2);
                        }
                        BleProduct.this.sendDataBroadcast("AA", substring, substring2);
                        z4 = true;
                    }
                    i3++;
                    upperCase = str3;
                    i = 8;
                    i2 = 0;
                }
                z = z4;
            } else if (TextUtils.isEmpty(upperCase) || !(upperCase.contains("CA") || upperCase.contains("CC"))) {
                z = false;
            } else {
                int indexOf = upperCase.indexOf("CA");
                int indexOf2 = upperCase.indexOf("CC");
                if (indexOf > -1 && indexOf2 > -1) {
                    indexOf = Math.min(indexOf, indexOf2);
                } else if (indexOf <= -1) {
                    indexOf = indexOf2 > -1 ? indexOf2 : 0;
                }
                upperCase = upperCase.substring(indexOf);
                int i4 = 8;
                String[] cut2 = StringUtils.cut(upperCase, 8);
                int i5 = 0;
                boolean z5 = false;
                while (i5 < cut2.length) {
                    if (cut2[i5].length() == i4) {
                        String upperCase2 = cut2[i5].substring(0, 2).toUpperCase();
                        if (!upperCase2.equals(str4) && !upperCase2.equals("CC")) {
                            return;
                        }
                        str = upperCase;
                        String substring3 = cut2[i5].substring(2, 4);
                        str2 = str4;
                        String substring4 = cut2[i5].substring(4, 6);
                        Integer.valueOf(substring3, 16).intValue();
                        int intValue2 = Integer.valueOf(substring4, 16).intValue();
                        substring3.hashCode();
                        switch (substring3.hashCode()) {
                            case 1536:
                                if (substring3.equals("00")) {
                                    c = 0;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1537:
                                if (substring3.equals(ModeDataVCUtil.HEAD_VERSION)) {
                                    c = 1;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1538:
                                if (substring3.equals("02")) {
                                    c = 2;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1539:
                                if (substring3.equals("03")) {
                                    c = 3;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1540:
                                if (substring3.equals("04")) {
                                    c = 4;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1541:
                                if (substring3.equals("05")) {
                                    c = 5;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1542:
                                if (substring3.equals("06")) {
                                    c = 6;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1543:
                                if (substring3.equals("07")) {
                                    c = 7;
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1544:
                                if (substring3.equals("08")) {
                                    c = '\b';
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1545:
                                if (substring3.equals("09")) {
                                    c = '\t';
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1546:
                            case 1547:
                            case 1548:
                            case 1549:
                            case 1550:
                            case 1551:
                            case 1552:
                            default:
                                c = 65535;
                                break;
                            case 1553:
                                if (substring3.equals("0A")) {
                                    c = '\n';
                                    break;
                                }
                                c = 65535;
                                break;
                            case 1554:
                                if (substring3.equals(Constans.piuCommand)) {
                                    c = 11;
                                    break;
                                }
                                c = 65535;
                                break;
                        }
                        switch (c) {
                            case 0:
                                strArr = cut2;
                                z2 = true;
                                if (BleProduct.this.mConnBleDevice != null) {
                                    BleProduct.this.mConnBleDevice.setBattery(intValue2);
                                    break;
                                }
                                break;
                            case 1:
                                strArr = cut2;
                                if (!"A0".equals(substring4)) {
                                    if (!"A1".equals(substring4)) {
                                        if (!"B0".equals(substring4)) {
                                            if (!"B1".equals(substring4)) {
                                                if (!"C0".equals(substring4)) {
                                                    if ("C1".equals(substring4)) {
                                                        BleProduct.this.mConnBleDevice.setReversal(1);
                                                        z2 = true;
                                                        break;
                                                    }
                                                } else {
                                                    BleProduct.this.mConnBleDevice.setReversal(0);
                                                }
                                            } else {
                                                z3 = true;
                                                BleProduct.this.mConnBleDevice.setMusic(1);
                                            }
                                        } else {
                                            BleProduct.this.mConnBleDevice.setMusic(0);
                                        }
                                    } else {
                                        z3 = true;
                                        BleProduct.this.mConnBleDevice.setRunMode(1);
                                    }
                                    z2 = z3;
                                    break;
                                } else {
                                    BleProduct.this.mConnBleDevice.setRunMode(0);
                                }
                                z2 = true;
                                break;
                            case 2:
                                strArr = cut2;
                                if (BleProduct.this.mConnBleDevice != null) {
                                    BleProduct.this.mConnBleDevice.clearData();
                                }
                                z2 = true;
                                break;
                            case 3:
                                strArr = cut2;
                                BleProduct.this.mConnBleDevice.setSpeed(intValue2);
                                BleProduct.this.mConnBleDevice.setMode(0);
                                BleProduct.this.followerControlRuleSpeed(intValue2);
                                z2 = true;
                                break;
                            case 4:
                                strArr = cut2;
                                BleProduct.this.mConnBleDevice.setMode(intValue2);
                                z2 = true;
                                break;
                            case 5:
                                strArr = cut2;
                                BleProduct.this.mConnBleDevice.setVibration(intValue2);
                                BleProduct.this.mConnBleDevice.setVibrationMode(0);
                                BleProduct.this.followerControlRuleVibrate(intValue2);
                                z2 = true;
                                break;
                            case 6:
                                strArr = cut2;
                                BleProduct.this.mConnBleDevice.setVibrationMode(intValue2);
                                z2 = true;
                                break;
                            case 7:
                                strArr = cut2;
                                BleProduct.this.mConnBleDevice.setStartPosition(intValue2);
                                z2 = true;
                                break;
                            case '\b':
                                strArr = cut2;
                                BleProduct.this.mConnBleDevice.setStopPosition(intValue2);
                                z2 = true;
                                break;
                            case '\t':
                                strArr = cut2;
                                BleProduct.this.mConnBleDevice.setSmoothness(intValue2);
                                z2 = true;
                                break;
                            case '\n':
                                strArr = cut2;
                                LogUtil.e(BleProduct.this.mDeviceName + "Ble位置：" + intValue2);
                                BleProduct.this.mConnBleDevice.setPosition(intValue2);
                                z2 = true;
                                break;
                            case 11:
                                BleProduct.this.mConnBleDevice.setPiu(intValue2);
                                z3 = true;
                                if (intValue2 != 1) {
                                    strArr = cut2;
                                    z2 = z3;
                                    break;
                                } else {
                                    BleProduct.this.followerControlRuleOpenPiu();
                                }
                            default:
                                strArr = cut2;
                                z2 = true;
                                break;
                        }
                        if (BleProduct.this.mConnectionStateCallback != null) {
                            BleProduct.this.mConnectionStateCallback.onNotifyChange_v2(upperCase2, substring3, substring4);
                        }
                        BleProduct.this.sendDataBroadcast(upperCase2, substring3, substring4);
                        z5 = z2;
                    } else {
                        strArr = cut2;
                        str = upperCase;
                        str2 = str4;
                        this.tempReceiver = strArr[i5];
                        LogUtil.e(BleProduct.this.mDeviceName + "多余留到下一次：" + this.tempReceiver);
                    }
                    i5++;
                    upperCase = str;
                    str4 = str2;
                    cut2 = strArr;
                    i4 = 8;
                }
                z = z5;
            }
            if (z) {
                return;
            }
            LogUtil.e(BleProduct.this.mDeviceName + "接收错误 ,data=" + upperCase);
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onCharacteristicRead(BluetoothGatt bluetoothGatt, BluetoothGattCharacteristic bluetoothGattCharacteristic, int i) {
            super.onCharacteristicRead(bluetoothGatt, bluetoothGattCharacteristic, i);
            Constans.UUID_BLUETOOTH_CODE_NOTIFY.equals(bluetoothGattCharacteristic.getUuid());
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onCharacteristicWrite(BluetoothGatt bluetoothGatt, BluetoothGattCharacteristic bluetoothGattCharacteristic, int i) {
            super.onCharacteristicWrite(bluetoothGatt, bluetoothGattCharacteristic, i);
            LogUtil.e(BleProduct.this.mDeviceName + "写入回调 characteristic:" + bluetoothGattCharacteristic + ", status:" + i);
            if (i == 0 && Constans.UUID_BLUETOOTH_DATA_WRITH.equals(bluetoothGattCharacteristic.getUuid())) {
                String bytesToHex = HexUtil.bytesToHex(bluetoothGattCharacteristic.getValue());
                int i2 = 16;
                if (BleProduct.this.mConnBleDevice == null || !"1002".equals(BleProduct.this.mConnBleDevice.getDeviceCode())) {
                    if (TextUtils.isEmpty(bytesToHex) || !bytesToHex.contains("CC")) {
                        return;
                    }
                    LogUtil.e(BleProduct.this.mBluetoothDevice.getName() + "写入回调通知:" + bytesToHex);
                    String[] cut = StringUtils.cut(bytesToHex, 8);
                    for (int i3 = 0; i3 < cut.length; i3++) {
                        LogUtil.e(BleProduct.this.mDeviceName + "onCharacteristicWrite（） ,array[" + i3 + "]=" + cut[i3]);
                        if (!cut[i3].substring(0, 2).toUpperCase().equals("CC")) {
                            return;
                        }
                        String substring = cut[i3].substring(2, 4);
                        String substring2 = cut[i3].substring(4, 6);
                        try {
                            int intValue = Integer.valueOf(substring2, 16).intValue();
                            int hashCode = substring.hashCode();
                            if (hashCode != 1537) {
                                if (hashCode == 1553 && substring.equals("0A")) {
                                    BleProduct.this.mConnBleDevice.setWritePosition(intValue);
                                }
                            } else if (substring.equals(ModeDataVCUtil.HEAD_VERSION) && "A1".equals(substring2)) {
                                BleProduct.this.mConnBleDevice.setWritePosition(BleProduct.this.mConnBleDevice.getPosition());
                                BleProduct bleProduct = BleProduct.this;
                                bleProduct.writeStartPosition(bleProduct.mConnBleDevice.getStartPosition());
                                BleProduct bleProduct2 = BleProduct.this;
                                bleProduct2.writeSmooth(bleProduct2.mConnBleDevice.getSmoothness());
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    return;
                }
                LogUtil.e(BleProduct.this.mBluetoothDevice.getName() + "写入回调通知:" + bytesToHex);
                if (TextUtils.isEmpty(bytesToHex) || !bytesToHex.contains("AA")) {
                    return;
                }
                String[] cut2 = StringUtils.cut(bytesToHex, 8);
                int i4 = 0;
                while (i4 < cut2.length) {
                    if (cut2[i4].length() == 8) {
                        LogUtil.e(BleProduct.this.mDeviceName + "onCharacteristicWrite（） ,array[" + i4 + "]=" + cut2[i4]);
                        if (!cut2[i4].substring(0, 2).toUpperCase().equals("AA")) {
                            return;
                        }
                        String substring3 = cut2[i4].substring(2, 4);
                        String substring4 = cut2[i4].substring(4, 6);
                        cut2[i4].substring(6, 8);
                        try {
                            int intValue2 = Integer.valueOf(substring4, i2).intValue();
                            int hashCode2 = substring3.hashCode();
                            if (hashCode2 != 1538) {
                                if (hashCode2 == 1540 && substring3.equals("04")) {
                                    if (BleProduct.this.mConnBleDevice != null) {
                                        BleProduct.this.mConnBleDevice.setSpeed(intValue2);
                                        BleProduct.this.mConnBleDevice.setMode(0);
                                    }
                                    BleProduct.this.followerControlRuleSpeed(intValue2);
                                    if (BleProduct.this.mConnectionStateCallback != null) {
                                        BleProduct.this.mConnectionStateCallback.onNotifyChange_v2("AA", substring3, substring4);
                                    }
                                    BleProduct.this.sendDataBroadcast("AA", substring3, substring4);
                                }
                            } else if (substring3.equals("02")) {
                                if (BleProduct.this.mConnBleDevice != null) {
                                    BleProduct.this.mConnBleDevice.clearData();
                                }
                                if (BleProduct.this.mConnectionStateCallback != null) {
                                    BleProduct.this.mConnectionStateCallback.onNotifyChange_v2("AA", substring3, substring4);
                                }
                                BleProduct.this.sendDataBroadcast("AA", substring3, substring4);
                            }
                        } catch (Exception e2) {
                            e2.printStackTrace();
                        }
                    }
                    i4++;
                    i2 = 16;
                }
            }
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onConnectionStateChange(BluetoothGatt bluetoothGatt, int i, int i2) {
            super.onConnectionStateChange(bluetoothGatt, i, i2);
            LogUtil.e(BleProduct.this.mDeviceName + "蓝牙onConnectionStateChange status：" + i + ", newState:" + i2);
            if (i != 0) {
                BleProduct.this.mStatus = 0;
                if (BleProduct.this.mConnectionStateCallback != null) {
                    BleProduct.this.mConnectionStateCallback.onProductCallback(bluetoothGatt.getDevice(), null);
                }
                if (BleProduct.this.mConnectionStateCallback != null) {
                    BleProduct.this.mConnectionStateCallback.onDisConnected(bluetoothGatt.getDevice());
                    return;
                }
                return;
            }
            if (i2 != 2) {
                if (i2 == 0) {
                    BleProduct.this.mStatus = 0;
                    if (BleProduct.this.mConnectionStateCallback != null) {
                        BleProduct.this.mConnectionStateCallback.onProductCallback(bluetoothGatt.getDevice(), null);
                    }
                    if (BleProduct.this.mConnectionStateCallback != null) {
                        BleProduct.this.mConnectionStateCallback.onDisConnected(bluetoothGatt.getDevice());
                        return;
                    }
                    return;
                }
                return;
            }
            BleProduct.this.mStatus = 2;
            BleProduct.this.initProductModel();
            if (BleProduct.this.mProductBaseControl == null) {
                BleProduct.this.mProductBaseControl = new ProductAAControl(BleProduct.this.mContext, BleProduct.this.mBluetoothGatt);
            }
            if (BleProduct.this.mConnBleDevice == null) {
                BleProduct.this.mConnBleDevice = new BleDeviceModel();
            }
            BleProduct.this.mConnBleDevice.setDeviceName(BleProduct.this.mDeviceName);
            BleProduct.this.mConnBleDevice.setDeviceAddress(bluetoothGatt.getDevice().getAddress());
            BleProduct.this.mConnBleDevice.setBluetoothDevice(bluetoothGatt.getDevice());
            BleProduct.this.mConnBleDevice.setSpeed(0);
            BleProduct.this.mConnBleDevice.setMode(0);
            BleProduct.this.mConnBleDevice.setVibration(0);
            BleProduct.this.mConnBleDevice.setVibrationMode(0);
            BleProduct.this.mConnBleDevice.setMusic(0);
            BleProduct.this.mConnBleDevice.setRunMode(0);
            BleProduct.this.mConnBleDevice.setPosition(0);
            BleProduct.this.mConnBleDevice.setSmoothness(0);
            BleProduct.this.mConnBleDevice.setDeviceCode(BleProduct.this.mDeviceCode);
            bluetoothGatt.requestConnectionPriority(1);
            bluetoothGatt.requestMtu(256);
            if (BleProduct.this.mConnectionStateCallback != null) {
                BleProduct.this.mConnectionStateCallback.onConnected(bluetoothGatt.getDevice(), BleProduct.this.mDeviceCode, BleProduct.this.mProductModel);
            }
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onDescriptorRead(BluetoothGatt bluetoothGatt, BluetoothGattDescriptor bluetoothGattDescriptor, int i) {
            super.onDescriptorRead(bluetoothGatt, bluetoothGattDescriptor, i);
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onDescriptorWrite(BluetoothGatt bluetoothGatt, BluetoothGattDescriptor bluetoothGattDescriptor, int i) {
            super.onDescriptorWrite(bluetoothGatt, bluetoothGattDescriptor, i);
            BleProduct bleProduct = BleProduct.this;
            if (i == 0) {
                LogUtil.e(bleProduct.mDeviceName + "onDescriptorWrite().GATT_SUCCESS 蓝牙开始监听成功");
            } else {
                LogUtil.e(bleProduct.mDeviceName + "onDescriptorWrite() ELSE");
            }
            BleProduct.this.mStatus = 2;
            BleProduct.this.initWriteMessage();
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onMtuChanged(BluetoothGatt bluetoothGatt, int i, int i2) {
            super.onMtuChanged(bluetoothGatt, i, i2);
            BleProduct bleProduct = BleProduct.this;
            if (i2 == 0) {
                LogUtil.e(bleProduct.mDeviceName + "onMtuChanged success MTU = " + i);
            } else {
                LogUtil.e(bleProduct.mDeviceName + "onMtuChanged fail " + i);
            }
            bluetoothGatt.discoverServices();
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onServicesDiscovered(BluetoothGatt bluetoothGatt, int i) {
            super.onServicesDiscovered(bluetoothGatt, i);
            LogUtil.e(BleProduct.this.mDeviceName + "获取蓝牙服务");
            if (i == 0) {
                setBleNotification();
            }
        }
    };
    private RetrofitRequest baseRequest = new RetrofitRequest();

    public interface ConnectionStateCallback {
        void onConnected(BluetoothDevice bluetoothDevice, String str, ProductModel productModel);

        void onDisConnected(BluetoothDevice bluetoothDevice);

        void onNotifyChange_v2(String str, String str2, String str3);

        void onProductCallback(BluetoothDevice bluetoothDevice, ProductModel productModel);
    }

    public BleProduct(Context context, BluetoothDevice bluetoothDevice, String str, String str2, ConnectionStateCallback connectionStateCallback) {
        this.mConnBleDevice = new BleDeviceModel();
        this.mDeviceCode = "";
        this.mDeviceName = "";
        this.mContext = context;
        this.mBluetoothDevice = bluetoothDevice;
        this.mDeviceName = str;
        this.mDeviceCode = str2;
        this.mConnectionStateCallback = connectionStateCallback;
        this.mConnBleDevice = new BleDeviceModel();
    }

    private void allProducts() {
        this.baseRequest.getApi().allProducts_v1().subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ProductsModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.bluetooth.BleProduct.5
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ProductsModel productsModel) {
                if (productsModel.getSuccess() == 1) {
                    SharedPreferencesUtil.put(BleProduct.this.mContext, Constans.ProductAllData, new Gson().toJson(productsModel));
                    try {
                        if (productsModel.getProducts() == null || productsModel.getProducts().length <= 0) {
                            return;
                        }
                        for (int i = 0; i < productsModel.getProducts().length; i++) {
                            RetrofitApi.getInstance(BleProduct.this.mContext).presenterSVGFile(productsModel.getProducts()[i]);
                            if (productsModel.getProducts()[i].getCode().equals(BleProduct.this.mDeviceCode)) {
                                BleProduct bleProduct = BleProduct.this;
                                bleProduct.mProductBaseControl = Factory.createProduct(bleProduct.mContext, BleProduct.this.mBluetoothGatt, productsModel.getProducts()[i]);
                                BleProduct.this.setmProductModel(productsModel.getProducts()[i]);
                                BleProduct.this.mConnBleDevice.setProductModel(BleProduct.this.mProductModel);
                                if (BleProduct.this.mConnectionStateCallback == null || BleProduct.this.mProductModel == null) {
                                    return;
                                }
                                BleProduct.this.mConnectionStateCallback.onProductCallback(BleProduct.this.mBluetoothDevice, BleProduct.this.mProductModel);
                                return;
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    private synchronized void closeBluetoothGatt() {
        BluetoothGatt bluetoothGatt = this.mBluetoothGatt;
        if (bluetoothGatt != null) {
            bluetoothGatt.close();
        }
    }

    private void followerControlRuleClosePiu() {
        if (getBluetoothDevice().getAddress().equals(MyApplication.getApplication().getDeviceAddress())) {
            for (Map.Entry<String, BleProduct> entry : BleManager.getInstance(this.mContext).getBleProducts().entrySet()) {
                if (!getBluetoothDevice().getAddress().equals(entry.getKey())) {
                    ControlRuleManager.getInstance().applyRuleClosePiu(entry.getKey());
                }
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void followerControlRuleOpenPiu() {
        if (getBluetoothDevice().getAddress().equals(MyApplication.getApplication().getDeviceAddress())) {
            for (Map.Entry<String, BleProduct> entry : BleManager.getInstance(this.mContext).getBleProducts().entrySet()) {
                if (!getBluetoothDevice().getAddress().equals(entry.getKey())) {
                    ControlRuleManager.getInstance().applyRuleOpenPiu(entry.getKey());
                }
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void followerControlRuleSpeed(int i) {
        if (getBluetoothDevice().getAddress().equals(MyApplication.getApplication().getDeviceAddress())) {
            for (Map.Entry<String, BleProduct> entry : BleManager.getInstance(this.mContext).getBleProducts().entrySet()) {
                if (!getBluetoothDevice().getAddress().equals(entry.getKey())) {
                    ControlRuleManager.getInstance().applyRuleToSpeed(entry.getKey(), i);
                }
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void followerControlRuleVibrate(int i) {
        if (getBluetoothDevice().getAddress().equals(MyApplication.getApplication().getDeviceAddress())) {
            for (Map.Entry<String, BleProduct> entry : BleManager.getInstance(this.mContext).getBleProducts().entrySet()) {
                if (!getBluetoothDevice().getAddress().equals(entry.getKey())) {
                    ControlRuleManager.getInstance().applyRuleToVibrate(entry.getKey(), i);
                }
            }
        }
    }

    private void getProduct(String str) {
        String str2 = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        if (TextUtils.isEmpty(str) || TextUtils.isEmpty(str2)) {
            return;
        }
        this.baseRequest.getApi().getProduct(str2, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ProductModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.bluetooth.BleProduct.6
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ProductModel productModel) {
                if (productModel.getSuccess() == 1) {
                    try {
                        RetrofitApi.getInstance(BleProduct.this.mContext).presenterSVGFile(productModel);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initProductModel() {
        ProductModel productModel;
        ProductModel productModel2;
        try {
            String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.ProductAllData, "");
            if (TextUtils.isEmpty(str)) {
                allProducts();
                ProductModel product = ProductManager.getProduct(this.mDeviceCode);
                if (product != null) {
                    this.mProductBaseControl = Factory.createProduct(this.mContext, this.mBluetoothGatt, product);
                    setmProductModel(product);
                    this.mConnBleDevice.setProductModel(this.mProductModel);
                    ConnectionStateCallback connectionStateCallback = this.mConnectionStateCallback;
                    if (connectionStateCallback == null || (productModel = this.mProductModel) == null) {
                        return;
                    }
                    connectionStateCallback.onProductCallback(this.mBluetoothDevice, productModel);
                    return;
                }
                return;
            }
            ProductsModel productsModel = (ProductsModel) new Gson().fromJson(str, ProductsModel.class);
            if (productsModel != null && productsModel.getProducts() != null && productsModel.getProducts().length > 0) {
                int i = 0;
                while (true) {
                    if (i >= productsModel.getProducts().length) {
                        break;
                    }
                    if (productsModel.getProducts()[i].getCode().equals(this.mDeviceCode)) {
                        this.mProductBaseControl = Factory.createProduct(this.mContext, this.mBluetoothGatt, productsModel.getProducts()[i]);
                        setmProductModel(productsModel.getProducts()[i]);
                        this.mConnBleDevice.setProductModel(this.mProductModel);
                        ConnectionStateCallback connectionStateCallback2 = this.mConnectionStateCallback;
                        if (connectionStateCallback2 != null) {
                            connectionStateCallback2.onProductCallback(this.mBluetoothDevice, this.mProductModel);
                        }
                    } else {
                        i++;
                    }
                }
            }
            if (this.mProductModel == null) {
                setmProductModel(ProductManager.getProduct(this.mDeviceCode));
                this.mConnBleDevice.setProductModel(this.mProductModel);
                ConnectionStateCallback connectionStateCallback3 = this.mConnectionStateCallback;
                if (connectionStateCallback3 == null || (productModel2 = this.mProductModel) == null) {
                    return;
                }
                connectionStateCallback3.onProductCallback(this.mBluetoothDevice, productModel2);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initServiceAndChara() {
        for (BluetoothGattService bluetoothGattService : this.mBluetoothGatt.getServices()) {
            for (BluetoothGattCharacteristic bluetoothGattCharacteristic : bluetoothGattService.getCharacteristics()) {
                int properties = bluetoothGattCharacteristic.getProperties();
                if ((properties & 2) > 0) {
                    LogUtil.w(this.mDeviceName + "read_chara=" + bluetoothGattCharacteristic.getUuid() + "----read_service=" + bluetoothGattService.getUuid());
                }
                if ((properties & 8) > 0) {
                    LogUtil.w(this.mDeviceName + "write_chara=" + bluetoothGattCharacteristic.getUuid() + "----write_service=" + bluetoothGattService.getUuid());
                }
                if ((properties & 4) > 0) {
                    LogUtil.w(this.mDeviceName + "write_chara=" + bluetoothGattCharacteristic.getUuid() + "----write_service=" + bluetoothGattService.getUuid());
                }
                if ((properties & 16) > 0) {
                    LogUtil.w(this.mDeviceName + "notify_chara=" + bluetoothGattCharacteristic.getUuid() + "----notify_service=" + bluetoothGattService.getUuid());
                }
                if ((properties & 32) > 0) {
                    LogUtil.w(this.mDeviceName + "indicate_chara=" + bluetoothGattCharacteristic.getUuid() + "----indicate_service=" + bluetoothGattService.getUuid());
                }
            }
        }
    }

    private synchronized void refreshDeviceCache() {
        BluetoothGatt bluetoothGatt;
        try {
            Method method = BluetoothGatt.class.getMethod("refresh", new Class[0]);
            if (method != null && (bluetoothGatt = this.mBluetoothGatt) != null) {
                LogUtil.e(this.mDeviceName + "refreshDeviceCache, is success:  " + ((Boolean) method.invoke(bluetoothGatt, new Object[0])).booleanValue());
            }
        } catch (Exception e) {
            LogUtil.e(this.mDeviceName + "exception occur while refreshing device: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private int valueScale(int i, double d) {
        int max = Math.max(Math.min((int) (i * d), 100), -100);
        return max < 0 ? max + 100 : max;
    }

    public void clearData() {
        this.mConnBleDevice.clearData();
    }

    public void connectGatt() {
        this.mBluetoothGatt = this.mBluetoothDevice.connectGatt(this.mContext, false, this.mGattCallback, 2);
        this.mStatus = 1;
    }

    public synchronized void destroy() {
        refreshDeviceCache();
        closeBluetoothGatt();
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null && !productBaseControl.isDisposed()) {
            this.mProductBaseControl.dispose();
        }
    }

    public synchronized void disconnectGatt() {
        BluetoothGatt bluetoothGatt = this.mBluetoothGatt;
        if (bluetoothGatt != null) {
            bluetoothGatt.disconnect();
        }
    }

    public BluetoothDevice getBluetoothDevice() {
        return this.mBluetoothDevice;
    }

    public BleDeviceModel getConnBleDevice() {
        return this.mConnBleDevice;
    }

    public String getDeviceCode() {
        return this.mDeviceCode;
    }

    public String getDeviceName() {
        return this.mDeviceName;
    }

    public ProductModel getmProductModel() {
        return this.mProductModel;
    }

    public int getmStatus() {
        return this.mStatus;
    }

    /* JADX WARN: Type inference failed for: r0v1, types: [com.hismith.hismithapp.bluetooth.BleProduct$4] */
    /* JADX WARN: Type inference failed for: r0v2, types: [com.hismith.hismithapp.bluetooth.BleProduct$3] */
    public void initWriteMessage() {
        try {
            if (this.mProductModel != null) {
                new Thread() { // from class: com.hismith.hismithapp.bluetooth.BleProduct.3
                    @Override // java.lang.Thread, java.lang.Runnable
                    public void run() {
                        super.run();
                        try {
                            LogUtil.e(BleProduct.this.mDeviceName + "蓝牙设备网络类型：" + BleProduct.this.mProductModel.getDeviceType());
                            boolean equals = BleProduct.this.mProductModel.getCommand().equals("CC");
                            BleProduct bleProduct = BleProduct.this;
                            if (equals) {
                                bleProduct.writeGetStatus();
                                Thread.sleep(1000L);
                                BleProduct.this.writeGetBattery();
                                return;
                            }
                            String deviceType = bleProduct.mProductModel.getDeviceType();
                            int hashCode = deviceType.hashCode();
                            if (hashCode == 70) {
                                if (deviceType.equals(Constans.DEVICE_TYPE_FUN)) {
                                    Thread.sleep(100L);
                                    BleProduct.this.writeGetBattery();
                                    return;
                                }
                                return;
                            }
                            if (hashCode == 2650) {
                                if (deviceType.equals(Constans.DEVICE_TYPE_SM)) {
                                    BleProduct.this.writeGetStatus();
                                    Thread.sleep(100L);
                                    BleProduct.this.writeGetSpeed();
                                    Thread.sleep(100L);
                                    BleProduct.this.writeGetMode();
                                    Thread.sleep(100L);
                                    return;
                                }
                                return;
                            }
                            if (hashCode == 82200 && deviceType.equals(Constans.DEVICE_TYPE_SM2)) {
                                BleProduct.this.writeGetStatus();
                                Thread.sleep(100L);
                                BleProduct.this.writeGetSpeed();
                                Thread.sleep(100L);
                                BleProduct.this.writeGetMode();
                                Thread.sleep(100L);
                                BleProduct.this.writeGetVibrate();
                                Thread.sleep(100L);
                                if (BleProduct.this.mProductModel.getMusic() == 1) {
                                    BleProduct.this.writeGetVoice();
                                }
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }.start();
            } else {
                new Thread() { // from class: com.hismith.hismithapp.bluetooth.BleProduct.4
                    @Override // java.lang.Thread, java.lang.Runnable
                    public void run() {
                        super.run();
                        try {
                            LogUtil.e(BleProduct.this.mDeviceName + "蓝牙设备本地设备号类型：" + BleProduct.this.mDeviceCode);
                            if (!"1001".equals(BleProduct.this.mDeviceCode) && !"1002".equals(BleProduct.this.mDeviceCode) && !"1003".equals(BleProduct.this.mDeviceCode) && !"3001".equals(BleProduct.this.mDeviceCode) && !"2001".equals(BleProduct.this.mDeviceCode)) {
                                BleProduct.this.writeGetStatus();
                                Thread.sleep(100L);
                                BleProduct.this.writeGetBattery();
                                return;
                            }
                            BleProduct.this.writeGetStatus();
                            Thread.sleep(100L);
                            BleProduct.this.writeGetSpeed();
                            Thread.sleep(100L);
                            BleProduct.this.writeGetMode();
                            Thread.sleep(100L);
                            if ("2001".equals(BleProduct.this.mDeviceCode)) {
                                BleProduct.this.writeGetVibrate();
                                Thread.sleep(100L);
                                BleProduct.this.writeGetVoice();
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }.start();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void sendDataBroadcast(String str, String str2, String str3) {
        Intent intent = new Intent(Constans.RECEIVER_BLE_RECE_DATA);
        intent.putExtra("Type", str);
        intent.putExtra(Constans.BleType, str);
        intent.putExtra(Constans.BleData0, str2);
        intent.putExtra(Constans.BleData1, str3);
        intent.putExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS, this.mBluetoothDevice.getAddress());
        this.mContext.sendBroadcast(intent);
    }

    public void setmDeviceName(String str) {
        this.mDeviceName = str;
    }

    public void setmProductModel(ProductModel productModel) {
        this.mProductModel = productModel;
    }

    public void setmStatus(int i) {
        this.mStatus = i;
    }

    public String toString() {
        return "BleProduct{mBluetoothDeviceName=" + this.mDeviceName + ", deviceAddress=" + this.mBluetoothDevice.getAddress() + ", mDeviceCode='" + this.mDeviceCode + "'}";
    }

    public void writeClosePiu() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeClosePiu();
        }
    }

    public void writeCloseReversal() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeCloseReversal();
        }
    }

    public void writeCloseVibrate() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeCloseVibrate();
        }
    }

    public void writeCloseVoice() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeCloseVoice();
        }
    }

    public void writeData(String str) {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeData(str);
        }
    }

    public void writeGetBattery() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeGetBattery();
        }
    }

    public void writeGetDeviceCode() {
        BluetoothGattService service;
        BluetoothGattCharacteristic characteristic;
        BluetoothGatt bluetoothGatt = this.mBluetoothGatt;
        if (bluetoothGatt == null || (service = bluetoothGatt.getService(Constans.UUID_BLUETOOTH_NAME)) == null || (characteristic = service.getCharacteristic(Constans.UUID_BLUETOOTH_CODE_NOTIFY)) == null || (characteristic.getProperties() & 2) <= 0) {
            return;
        }
        LogUtil.e(this.mDeviceName + "发送获取产品识别码：" + this.mBluetoothGatt.readCharacteristic(characteristic));
    }

    public void writeGetMode() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeGetMode();
        }
    }

    public void writeGetSpeed() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeGetSpeed();
        }
    }

    public void writeGetStatus() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeGetStatus();
        }
    }

    public void writeGetVibrate() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeGetVibrate();
        }
    }

    public void writeGetVoice() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeGetVoice();
        }
    }

    public void writeModeLine1(int i) {
        LogUtil.e(this.mDeviceName + "Ble发送ModeLine1:" + i);
        ProductModel productModel = this.mProductModel;
        if (productModel == null || productModel.getDeviceType() == null || !this.mProductModel.getDeviceType().equals(Constans.DEVICE_TYPE_SMS)) {
            ProductBaseControl productBaseControl = this.mProductBaseControl;
            if (productBaseControl != null) {
                productBaseControl.writeSpeed(i);
                return;
            }
            return;
        }
        ProductBaseControl productBaseControl2 = this.mProductBaseControl;
        if (productBaseControl2 != null) {
            productBaseControl2.writePosition(i);
        }
    }

    public void writeModeLine2(int i) {
        ProductBaseControl productBaseControl;
        ProductModel productModel = this.mProductModel;
        if ((productModel == null || productModel.getDeviceType() == null || !this.mProductModel.getDeviceType().equals(Constans.DEVICE_TYPE_SMS)) && (productBaseControl = this.mProductBaseControl) != null) {
            productBaseControl.writeVibrate(i);
        }
    }

    public void writeName(String str) {
        BluetoothGattService service;
        BluetoothGattCharacteristic characteristic;
        BluetoothGatt bluetoothGatt = this.mBluetoothGatt;
        if (bluetoothGatt == null || (service = bluetoothGatt.getService(Constans.UUID_BLUETOOTH_NAME)) == null || (characteristic = service.getCharacteristic(Constans.UUID_BLUETOOTH_NAME_WRITE)) == null) {
            return;
        }
        byte[] bytes = str.getBytes(StandardCharsets.UTF_8);
        LogUtil.e(this.mDeviceName + "蓝牙名称：" + str + ", " + bytes.length + ", " + Arrays.toString(bytes));
        if (bytes.length >= 20) {
            LogUtil.e(this.mDeviceName + "修改蓝牙名称超长");
        } else {
            LogUtil.e(this.mDeviceName + "发送修改蓝牙名称:" + characteristic.setValue(bytes) + ", " + this.mBluetoothGatt.writeCharacteristic(characteristic));
        }
    }

    public void writeName(byte[] bArr) {
        BluetoothGattService service;
        BluetoothGattCharacteristic characteristic;
        BluetoothGatt bluetoothGatt = this.mBluetoothGatt;
        if (bluetoothGatt == null || (service = bluetoothGatt.getService(Constans.UUID_BLUETOOTH_NAME)) == null || (characteristic = service.getCharacteristic(Constans.UUID_BLUETOOTH_NAME_WRITE)) == null) {
            return;
        }
        if (bArr.length >= 20) {
            LogUtil.e(this.mDeviceName + "修改蓝牙名称超长");
        } else {
            LogUtil.e(this.mDeviceName + "发送修改蓝牙名称:" + characteristic.setValue(bArr) + ", " + this.mBluetoothGatt.writeCharacteristic(characteristic));
        }
    }

    public void writeOpenClimax() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeOpenClimax();
        }
    }

    public void writeOpenClimax01() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeOpenClimax01();
        }
    }

    public void writeOpenPiu() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeOpenPiu();
        }
    }

    public void writeOpenReversal() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeOpenReversal();
        }
    }

    public void writeOpenVoice() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeOpenVoice();
        }
    }

    public void writePosition(int i) {
        LogUtil.e(this.mDeviceName + "Ble发送位置：" + i);
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writePosition(i);
        }
    }

    public void writeRunAuto() {
        LogUtil.e(this.mDeviceName + "Ble发送自动模式");
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeRunAuto();
        }
    }

    public void writeRunManual() {
        LogUtil.e(this.mDeviceName + "Ble发送手动模式");
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeRunManual();
        }
    }

    public void writeShutdown() {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeShutdown();
            BleDeviceModel bleDeviceModel = this.mConnBleDevice;
            if (bleDeviceModel != null) {
                bleDeviceModel.clearData();
            }
        }
    }

    public void writeSmooth(int i) {
        LogUtil.e(this.mDeviceName + "Ble发送顺滑度：" + i);
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeSmooth(i);
        }
    }

    public void writeSpeed(int i) {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeSpeed(i);
        }
    }

    public void writeSpeedMode(int i) {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeSpeedMode(i);
        }
    }

    public void writeStartPosition(int i) {
        LogUtil.e(this.mDeviceName + "Ble发送开始位置：" + i);
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeStartPosition(i);
        }
    }

    public void writeStopPosition(int i) {
        LogUtil.e(this.mDeviceName + "Ble发送结束位置：" + i);
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeStopPosition(i);
        }
    }

    public void writeVibrate(int i) {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeVibrate(i);
        }
    }

    public void writeVibrateMode(int i) {
        ProductBaseControl productBaseControl = this.mProductBaseControl;
        if (productBaseControl != null) {
            productBaseControl.writeVibrateMode(i);
        }
    }
}
