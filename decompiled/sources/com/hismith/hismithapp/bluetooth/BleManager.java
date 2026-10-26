package com.hismith.hismithapp.bluetooth;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import androidx.media3.exoplayer.ExoPlayer;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.BleUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HexUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class BleManager {
    public static final String BLE_STATE = "ble_state";
    private static List<String> autoAddress;
    private static BleManager instance;
    private static BluetoothAdapter mBluetoothAdapter;
    private static BluetoothLeScanner mBluetoothLeScanner;
    private static BluetoothManager mBluetoothManger;
    private Context mContext;
    private OnConnectStateListener onConnectStateListener;
    private OnNotifyChangeListener_v2 onNotifyChangeListener_v2;
    private OnScanDeviceListener onScanDeviceListener;
    private long scanDeviceTime;
    private int mStatus = 0;
    private boolean isConnecting = false;
    private boolean isWriteSpeed = true;
    private int maxConnectCount = 5;
    private int mScanStatus = 0;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.bluetooth.BleManager.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            try {
                int i = message.what;
                if (i != 0) {
                    if (i != 1) {
                        if (i == 5) {
                            BleManager.this.stopLeScan();
                            BleManager.this.mHandler.sendEmptyMessageDelayed(6, ExoPlayer.DEFAULT_DETACH_SURFACE_TIMEOUT_MS);
                        } else if (i == 6) {
                            BleManager.this.startLeScan();
                        }
                    } else if (BleManager.this.mContext != null) {
                        ToastUtil.showShortToastCenter(BleManager.this.mContext, ((String) message.obj) + " " + LanguageUtil.getLanguageByKey(BleManager.this.mContext, "bluetoothConnected"));
                    }
                } else if (BleManager.this.mContext != null) {
                    ToastUtil.showShortToastCenter(BleManager.this.mContext, ((String) message.obj) + " " + LanguageUtil.getLanguageByKey(BleManager.this.mContext, "bluetoothDisconnected"));
                }
                return false;
            } catch (Exception e) {
                e.printStackTrace();
                return false;
            }
        }
    });
    private boolean isRunning = false;
    Runnable runnable = new Runnable() { // from class: com.hismith.hismithapp.bluetooth.BleManager.2
        @Override // java.lang.Runnable
        public void run() {
            while (BleManager.this.isRunning) {
                if (System.currentTimeMillis() - BleManager.this.scanDeviceTime > ExoPlayer.DEFAULT_DETACH_SURFACE_TIMEOUT_MS) {
                    BleManager.this.mHandler.sendEmptyMessage(5);
                    BleManager.this.isRunning = false;
                }
                try {
                    Thread.sleep(1000L);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    };
    private final ScanCallback scanCallback = new ScanCallback() { // from class: com.hismith.hismithapp.bluetooth.BleManager.3
        @Override // android.bluetooth.le.ScanCallback
        public void onScanResult(int i, ScanResult scanResult) {
            String bytesToHex;
            int indexOf;
            super.onScanResult(i, scanResult);
            BleManager.this.scanDeviceTime = System.currentTimeMillis();
            LogUtil.i("5.0搜索， " + scanResult.getDevice().getName() + ", address:" + scanResult.getDevice().getAddress());
            if (scanResult.getScanRecord() == null || (indexOf = (bytesToHex = HexUtil.bytesToHex(scanResult.getScanRecord().getBytes())).indexOf("4849534D495448")) == -1) {
                return;
            }
            String substring = indexOf > 4 ? bytesToHex.substring(indexOf - 4, indexOf) : "";
            String name = BleUtil.parseAdertisedData(scanResult.getScanRecord().getBytes()).getName();
            if (TextUtils.isEmpty(name)) {
                name = scanResult.getDevice().getName();
            }
            LogUtil.w("5.0搜索， name:" + scanResult.getDevice().getName() + ", name2:" + name + ", code:" + substring + ", address:" + scanResult.getDevice().getAddress() + ", 信号强度：" + scanResult.getRssi() + ", scan:" + bytesToHex);
            String address = scanResult.getDevice().getAddress();
            if (BleManager.this.onScanDeviceListener != null) {
                BleManager.this.onScanDeviceListener.onScanDevice(scanResult.getDevice(), name, substring);
            }
            if (BleManager.autoAddress.contains(address) && !BleManager.this.isConnecting) {
                BleManager.this.isConnecting = true;
                BleManager.this.connectGatt(scanResult.getDevice(), name, substring);
            } else if (BleManager.this.isConnecting) {
                LogUtil.e("有设备正在连接，请稍后：" + name);
            }
        }
    };
    BluetoothAdapter.LeScanCallback leScanCallback = new BluetoothAdapter.LeScanCallback() { // from class: com.hismith.hismithapp.bluetooth.BleManager.4
        @Override // android.bluetooth.BluetoothAdapter.LeScanCallback
        public void onLeScan(BluetoothDevice bluetoothDevice, int i, byte[] bArr) {
            BleManager.this.scanDeviceTime = System.currentTimeMillis();
            String bytesToHex = HexUtil.bytesToHex(bArr);
            int indexOf = bytesToHex.indexOf("4849534D495448");
            if (indexOf != -1) {
                String substring = indexOf > 4 ? bytesToHex.substring(indexOf - 4, indexOf) : "";
                String name = bluetoothDevice.getName();
                try {
                    name = BleAdParse.parseScanRecodeData(bArr).localName;
                } catch (Exception e) {
                    e.printStackTrace();
                }
                LogUtil.i("BleManager LeScanCallback蓝牙设备 name:" + bluetoothDevice.getName() + ", name2:" + name + ", address:" + bluetoothDevice.getAddress() + ", Code:" + substring + ", ScanRecord:" + bytesToHex + ", type:" + bluetoothDevice.getType());
                String address = bluetoothDevice.getAddress();
                if (BleManager.this.onScanDeviceListener != null) {
                    BleManager.this.onScanDeviceListener.onScanDevice(bluetoothDevice, name, substring);
                }
                if (BleManager.autoAddress.contains(address) && !BleManager.this.isConnecting) {
                    BleManager.this.isConnecting = true;
                    BleManager.this.connectGatt(bluetoothDevice, name, substring);
                } else if (BleManager.this.isConnecting) {
                    LogUtil.e("有设备正在连接，请稍后：" + name);
                }
            }
        }
    };
    private Map<String, BleProduct> bleProducts = new LinkedHashMap();
    private Map<String, BleProduct> bleMainProducts = new LinkedHashMap();
    private Map<String, BleProduct> bleFunProducts = new LinkedHashMap();

    public interface OnConnectStateListener {
        void onConnected(int i);

        void onDisconnected(int i);
    }

    public interface OnNotifyChangeListener_v2 {
        void onNotifyChange_v2(String str, String str2, String str3);
    }

    public interface OnScanDeviceListener {
        void onScanDevice(BluetoothDevice bluetoothDevice, String str, String str2);
    }

    public BleManager(Context context) {
        this.mContext = context;
        autoAddress = new ArrayList();
    }

    public static BleManager getInstance(Context context) {
        synchronized (BleManager.class) {
            if (instance == null) {
                instance = new BleManager(context);
                if (context.getPackageManager().hasSystemFeature("android.hardware.bluetooth_le")) {
                    BluetoothManager bluetoothManager = (BluetoothManager) context.getSystemService("bluetooth");
                    mBluetoothManger = bluetoothManager;
                    BluetoothAdapter adapter = bluetoothManager.getAdapter();
                    mBluetoothAdapter = adapter;
                    if (adapter != null) {
                        mBluetoothLeScanner = mBluetoothAdapter.getBluetoothLeScanner();
                        autoAddress = new ArrayList(new HashSet(SharedPreferencesUtil.getStringList(context, Constans.bluetoothConnAddress)));
                        LogUtil.e("autoAddress:" + Arrays.toString(autoAddress.toArray()));
                    }
                }
            }
        }
        return instance;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initApplicationProduct() {
        for (Map.Entry<String, BleProduct> entry : this.bleProducts.entrySet()) {
            if (MyApplication.getApplication().getmProductModel() == null) {
                MyApplication.getApplication().setDeviceName(entry.getValue().getDeviceName());
                MyApplication.getApplication().setDeviceAddress(entry.getValue().getBluetoothDevice().getAddress());
                MyApplication.getApplication().setDeviceCode(entry.getValue().getDeviceCode());
                MyApplication.getApplication().setmProductModel(entry.getValue().getmProductModel());
                SharedPreferencesUtil.put(this.mContext, Constans.bluetoothConnName, entry.getValue().getDeviceName());
                ControlRuleManager.getInstance().setLeaderId(entry.getValue().getBluetoothDevice().getAddress());
                this.mStatus = 2;
                Intent intent = new Intent(Constans.RECEIVER_BLE_CONN_STATE);
                intent.putExtra("ble_state", 2);
                this.mContext.sendBroadcast(intent);
                LogUtil.e("RECEIVER_BLE_CODE_STATE");
                this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_BLE_CODE_STATE));
                this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_PRODUCT_INFO));
                return;
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendBroadcastDeviceState(String str, int i) {
        Intent intent = new Intent(Constans.RECEIVER_BLE_DEVICE_STATE);
        intent.putExtra(Constans.EXTRA_DEVICE_ADDRESS, str);
        intent.putExtra("ble_state", i);
        this.mContext.sendBroadcast(intent);
    }

    private void writeGetMode() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeGetMode();
        }
    }

    private void writeGetSpeed() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeGetSpeed();
        }
    }

    private void writeGetStatus() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeGetStatus();
        }
    }

    private void writeGetVoice() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeGetVoice();
        }
    }

    public void closeGatt(BluetoothDevice bluetoothDevice) {
        BleProduct remove;
        if (bluetoothDevice.getAddress().equals(MyApplication.getApplication().getDeviceAddress())) {
            MyApplication.getApplication().setDeviceCode("");
            MyApplication.getApplication().setDeviceName("");
            MyApplication.getApplication().setDeviceAddress("");
            ControlRuleManager.getInstance().clearLeader();
            MyApplication.getApplication().setmProductModel(null);
        }
        if (!this.bleProducts.containsKey(bluetoothDevice.getAddress()) || (remove = this.bleProducts.remove(bluetoothDevice.getAddress())) == null) {
            return;
        }
        remove.destroy();
    }

    public synchronized void connectGatt(BluetoothDevice bluetoothDevice, final String str, String str2) {
        LogUtil.e("tttttr bleProducts.contains:" + this.bleProducts.containsKey(bluetoothDevice.getAddress()));
        if (bluetoothDevice != null && !this.bleProducts.containsKey(bluetoothDevice.getAddress()) && !isExceedMaxConnection()) {
            LogUtil.e(str + " 发起蓝牙连接, 设备号：" + str2);
            BleProduct bleProduct = new BleProduct(this.mContext, bluetoothDevice, str, str2, new BleProduct.ConnectionStateCallback() { // from class: com.hismith.hismithapp.bluetooth.BleManager.5
                @Override // com.hismith.hismithapp.bluetooth.BleProduct.ConnectionStateCallback
                public void onConnected(BluetoothDevice bluetoothDevice2, String str3, ProductModel productModel) {
                    LogUtil.e(str + " 连接成功");
                    SharedPreferencesUtil.put(BleManager.this.mContext, Constans.LAST_CONNECTED_DEVICE_TIME, Long.valueOf(System.currentTimeMillis()));
                    if (!BleManager.autoAddress.contains(bluetoothDevice2.getAddress())) {
                        BleManager.autoAddress.add(bluetoothDevice2.getAddress());
                        SharedPreferencesUtil.putStringList(BleManager.this.mContext, Constans.bluetoothConnAddress, BleManager.autoAddress);
                    }
                    if (BleManager.this.onConnectStateListener != null) {
                        BleManager.this.onConnectStateListener.onConnected(BleManager.this.bleProducts.size());
                    }
                    BleManager.this.sendBroadcastDeviceState(bluetoothDevice2.getAddress(), 2);
                    BleManager.this.initApplicationProduct();
                    LogUtil.e("BleManager onConnected setDeviceRule:" + str);
                    ControlRuleManager.getInstance().setDeviceRule(bluetoothDevice2.getAddress(), productModel);
                    Message message = new Message();
                    message.what = 1;
                    message.obj = str;
                    BleManager.this.mHandler.sendMessage(message);
                }

                @Override // com.hismith.hismithapp.bluetooth.BleProduct.ConnectionStateCallback
                public void onDisConnected(BluetoothDevice bluetoothDevice2) {
                    LogUtil.e(str + " 断开连接, isConnecting:" + BleManager.this.isConnecting);
                    BleManager.this.closeGatt(bluetoothDevice2);
                    if (BleManager.this.onConnectStateListener != null) {
                        BleManager.this.onConnectStateListener.onDisconnected(BleManager.this.bleProducts.size());
                    }
                    BleManager.this.sendBroadcastDeviceState(bluetoothDevice2.getAddress(), 0);
                    if (BleManager.this.bleProducts.size() == 0) {
                        BleManager.this.mStatus = 0;
                        MyApplication.getApplication().setDeviceName("");
                        MyApplication.getApplication().setDeviceAddress("");
                        ControlRuleManager.getInstance().clearLeader();
                        MyApplication.getApplication().setDeviceCode("");
                        MyApplication.getApplication().setmProductModel(null);
                        Intent intent = new Intent(Constans.RECEIVER_BLE_CONN_STATE);
                        intent.putExtra("ble_state", 0);
                        BleManager.this.mContext.sendBroadcast(intent);
                        SharedPreferencesUtil.put(BleManager.this.mContext, Constans.bluetoothConnName, "");
                    }
                    BleManager.this.initApplicationProduct();
                    Message message = new Message();
                    message.what = 0;
                    message.obj = str;
                    BleManager.this.mHandler.sendMessage(message);
                }

                @Override // com.hismith.hismithapp.bluetooth.BleProduct.ConnectionStateCallback
                public void onNotifyChange_v2(String str3, String str4, String str5) {
                    if (BleManager.this.onNotifyChangeListener_v2 != null) {
                        BleManager.this.onNotifyChangeListener_v2.onNotifyChange_v2(str3, str4, str5);
                    }
                }

                @Override // com.hismith.hismithapp.bluetooth.BleProduct.ConnectionStateCallback
                public void onProductCallback(BluetoothDevice bluetoothDevice2, ProductModel productModel) {
                    BleManager.this.isConnecting = false;
                    LogUtil.e("onProductCallback " + str + ", " + productModel + ", isConnecting:" + BleManager.this.isConnecting);
                    if (productModel == null) {
                        if (BleManager.this.bleProducts.get(bluetoothDevice2.getAddress()) == null || ((BleProduct) BleManager.this.bleProducts.get(bluetoothDevice2.getAddress())).getmProductModel() == null || !Constans.DEVICE_TYPE_FUN.equals(((BleProduct) BleManager.this.bleProducts.get(bluetoothDevice2.getAddress())).getmProductModel().getDeviceType())) {
                            return;
                        }
                        BleManager.this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_PRODUCT_INFO));
                        return;
                    }
                    if (MyApplication.getApplication().getDeviceAddress().equals(bluetoothDevice2.getAddress())) {
                        MyApplication.getApplication().setmProductModel(productModel);
                        BleManager.this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_PRODUCT_INFO));
                    } else if (Constans.DEVICE_TYPE_FUN.equals(productModel.getDeviceType())) {
                        BleManager.this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_PRODUCT_INFO));
                    }
                }
            });
            bleProduct.connectGatt();
            LogUtil.e("new bleProduct：" + bleProduct.toString());
            this.bleProducts.put(bluetoothDevice.getAddress(), bleProduct);
            LogUtil.e("tttttr bleProducts.size()" + this.bleProducts.size());
        } else if (isConnectedBleProduct(bluetoothDevice.getAddress()) == 2) {
            LogUtil.e(str + "已经连接");
            OnConnectStateListener onConnectStateListener = this.onConnectStateListener;
            if (onConnectStateListener != null) {
                onConnectStateListener.onConnected(this.bleProducts.size());
            }
        } else {
            LogUtil.e(str + " 正在连接中, isConnecting:" + this.isConnecting);
        }
    }

    public void disconnectAllGatt() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().disconnectGatt();
        }
    }

    public void disconnectGatt(BluetoothDevice bluetoothDevice, String str) {
        BleProduct bleProduct;
        autoAddress.remove(bluetoothDevice.getAddress());
        LogUtil.e("disconnectGatt(" + str + "), 剩余自动连接：" + Arrays.toString(autoAddress.toArray()));
        SharedPreferencesUtil.putStringList(this.mContext, Constans.bluetoothConnAddress, autoAddress);
        if (bluetoothDevice == null || (bleProduct = this.bleProducts.get(bluetoothDevice.getAddress())) == null) {
            return;
        }
        if (bleProduct.getmStatus() == 2) {
            bleProduct.disconnectGatt();
        } else {
            bleProduct.disconnectGatt();
            closeGatt(bluetoothDevice);
        }
    }

    public String getBleAAData(int i, int i2) {
        String hexString = Integer.toHexString(i);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        String hexString2 = Integer.toHexString(i2);
        if (hexString2.length() < 2) {
            hexString2 = "0" + hexString2;
        }
        String hexString3 = Integer.toHexString(i + i2);
        if (hexString3.length() < 2) {
            hexString3 = "0" + hexString3;
        }
        return "AA" + hexString + hexString2 + hexString3;
    }

    public String getBleCCData(int i, int i2) {
        String hexString = Integer.toHexString(i);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        String hexString2 = Integer.toHexString(i2);
        if (hexString2.length() < 2) {
            hexString2 = "0" + hexString2;
        }
        String hexString3 = Integer.toHexString(i + i2);
        if (hexString3.length() < 2) {
            hexString3 = "0" + hexString3;
        }
        return "CC" + hexString + hexString2 + hexString3;
    }

    public Map<String, BleProduct> getBleProducts() {
        return this.bleProducts;
    }

    public BluetoothAdapter getBluetoothAdapter() {
        return mBluetoothAdapter;
    }

    public BleProduct getConnBleProduct(String str) {
        if (TextUtils.isEmpty(str)) {
            return null;
        }
        return this.bleProducts.get(str);
    }

    public BleProduct getFirstFunBleProduct() {
        for (Map.Entry<String, BleProduct> entry : this.bleProducts.entrySet()) {
            if (entry.getValue().getmProductModel() != null && Constans.DEVICE_TYPE_FUN.equals(entry.getValue().getmProductModel().getDeviceType())) {
                return entry.getValue();
            }
        }
        return null;
    }

    public BleProduct getFirstNotFunBleProduct() {
        for (Map.Entry<String, BleProduct> entry : this.bleProducts.entrySet()) {
            if (entry.getValue().getmProductModel() != null && !Constans.DEVICE_TYPE_FUN.equals(entry.getValue().getmProductModel().getDeviceType())) {
                return entry.getValue();
            }
        }
        return null;
    }

    public Map<String, BleProduct> getFunBleProducts() {
        LinkedHashMap linkedHashMap = new LinkedHashMap();
        for (Map.Entry<String, BleProduct> entry : this.bleProducts.entrySet()) {
            if (entry.getValue() != null && entry.getValue().getmProductModel() != null && Constans.DEVICE_TYPE_FUN.equals(entry.getValue().getmProductModel().getDeviceType()) && entry.getValue().getmProductModel().getFuns() != null) {
                for (int i = 0; i < entry.getValue().getmProductModel().getFuns().length; i++) {
                    linkedHashMap.put(entry.getValue().getmProductModel().getFuns()[i].getfCommand(), entry.getValue());
                }
            }
        }
        return linkedHashMap;
    }

    public OnConnectStateListener getOnConnectStateListener() {
        return this.onConnectStateListener;
    }

    public OnNotifyChangeListener_v2 getOnNotifyChangeListener_v2() {
        return this.onNotifyChangeListener_v2;
    }

    public OnScanDeviceListener getOnScanDeviceListener() {
        return this.onScanDeviceListener;
    }

    public int getStatus() {
        return this.mStatus;
    }

    public BleDeviceModel getmConnBleDevice() {
        return this.bleProducts.containsKey(MyApplication.getApplication().getDeviceAddress()) ? this.bleProducts.get(MyApplication.getApplication().getDeviceAddress()).getConnBleDevice() : new BleDeviceModel();
    }

    public boolean hasPiuFunBle() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (true) {
            if (!it.hasNext()) {
                return false;
            }
            Map.Entry<String, BleProduct> next = it.next();
            if (next.getValue() != null && next.getValue().getmProductModel() != null && Constans.DEVICE_TYPE_FUN.equals(next.getValue().getmProductModel().getDeviceType()) && next.getValue().getmProductModel().getFuns() != null) {
                for (int i = 0; i < next.getValue().getmProductModel().getFuns().length; i++) {
                    if (Constans.piuCommand.equals(next.getValue().getmProductModel().getFuns()[i].getfCommand())) {
                        return true;
                    }
                }
            }
        }
    }

    public void initWriteMessage() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().initWriteMessage();
        }
    }

    public int isConnectedBleProduct(String str) {
        Map<String, BleProduct> map = this.bleProducts;
        if (map == null || !map.containsKey(str)) {
            return 0;
        }
        return this.bleProducts.get(str).getmStatus();
    }

    public boolean isExceedMaxConnection() {
        return this.bleProducts.size() >= this.maxConnectCount;
    }

    public boolean isWriteSpeed() {
        return this.isWriteSpeed;
    }

    public void release() {
        disconnectAllGatt();
        MyApplication.getApplication().setDeviceCode("");
        MyApplication.getApplication().setDeviceName("");
        MyApplication.getApplication().setDeviceAddress("");
        ControlRuleManager.getInstance().clearLeader();
        MyApplication.getApplication().setmProductModel(null);
        if (instance != null) {
            mBluetoothAdapter = null;
            mBluetoothManger = null;
            instance = null;
        }
    }

    public void selectApplicationProduct(String str) {
        boolean z = false;
        for (Map.Entry<String, BleProduct> entry : this.bleProducts.entrySet()) {
            if (entry.getKey().equals(str)) {
                MyApplication.getApplication().setDeviceName(entry.getValue().getDeviceName());
                MyApplication.getApplication().setDeviceAddress(entry.getValue().getBluetoothDevice().getAddress());
                MyApplication.getApplication().setDeviceCode(entry.getValue().getDeviceCode());
                MyApplication.getApplication().setmProductModel(entry.getValue().getmProductModel());
                SharedPreferencesUtil.put(this.mContext, Constans.bluetoothConnName, entry.getValue().getDeviceName());
                ControlRuleManager.getInstance().setLeaderId(entry.getValue().getBluetoothDevice().getAddress());
                z = true;
            }
        }
        if (z) {
            this.mStatus = 2;
            Intent intent = new Intent(Constans.RECEIVER_BLE_CONN_STATE);
            intent.putExtra("ble_state", 2);
            this.mContext.sendBroadcast(intent);
            LogUtil.e("RECEIVER_BLE_CODE_STATE");
            this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_BLE_CODE_STATE));
            this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_PRODUCT_INFO));
        }
    }

    public void setOnConnectStateListener(OnConnectStateListener onConnectStateListener) {
        this.onConnectStateListener = onConnectStateListener;
    }

    public void setOnNotifyChangeListener_v2(OnNotifyChangeListener_v2 onNotifyChangeListener_v2) {
        this.onNotifyChangeListener_v2 = onNotifyChangeListener_v2;
    }

    public void setOnScanDeviceListener(OnScanDeviceListener onScanDeviceListener) {
        this.onScanDeviceListener = onScanDeviceListener;
    }

    public void setStatus(int i) {
        this.mStatus = i;
    }

    public void setWriteSpeed(boolean z) {
        this.isWriteSpeed = z;
    }

    public void startLeScan() {
        boolean z = Build.VERSION.SDK_INT < 31 || this.mContext.getPackageManager().checkPermission("android.permission.BLUETOOTH_SCAN", this.mContext.getPackageName()) == 0;
        BluetoothAdapter bluetoothAdapter = mBluetoothAdapter;
        if (bluetoothAdapter == null || !bluetoothAdapter.isEnabled() || !z) {
            LogUtil.e("没有权限");
            return;
        }
        LogUtil.e("开始搜索");
        if (mBluetoothLeScanner != null) {
            ScanSettings.Builder scanMode = new ScanSettings.Builder().setScanMode(0);
            if (mBluetoothAdapter.isOffloadedScanBatchingSupported()) {
                scanMode.setReportDelay(0L);
            }
            mBluetoothLeScanner.startScan((List<ScanFilter>) null, scanMode.build(), this.scanCallback);
        } else {
            mBluetoothAdapter.startLeScan(this.leScanCallback);
        }
        this.mScanStatus = 1;
        this.scanDeviceTime = System.currentTimeMillis();
        this.isRunning = true;
        new Thread(this.runnable).start();
    }

    public void stopLeScan() {
        if (mBluetoothAdapter != null) {
            LogUtil.e("结束搜索");
            if (mBluetoothLeScanner != null) {
                mBluetoothLeScanner.stopScan(this.scanCallback);
            } else {
                mBluetoothAdapter.stopLeScan(this.leScanCallback);
            }
            this.mScanStatus = 2;
            this.isRunning = false;
        }
    }

    public void writeClosePiu() {
        if (!ProductUtil.isFPiuType()) {
            Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
            while (it.hasNext()) {
                it.next().getValue().writeClosePiu();
            }
        } else {
            BleProduct connBleProduct = getConnBleProduct(MyApplication.getApplication().getDeviceAddress());
            if (connBleProduct != null) {
                connBleProduct.writeClosePiu();
            }
        }
    }

    public void writeCloseReversal() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeCloseReversal();
        }
    }

    public void writeCloseVibrate() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeCloseVibrate();
        }
    }

    public void writeCloseVoice() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeCloseVoice();
        }
    }

    public void writeData(String str) {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeData(str);
        }
    }

    public void writeGetBattery() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeGetBattery();
        }
    }

    public void writeGetVibrate() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeGetVibrate();
        }
    }

    public void writeModeLine1(int i) {
        if (this.isWriteSpeed) {
            LogUtil.e("writeModeLine1：" + i);
            BleProduct connBleProduct = getConnBleProduct(MyApplication.getApplication().getDeviceAddress());
            if (connBleProduct != null) {
                connBleProduct.writeModeLine1(i);
            }
        }
    }

    public void writeModeLine2(int i) {
        if (this.isWriteSpeed) {
            LogUtil.e("writeModeLine2：" + i);
            BleProduct connBleProduct = getConnBleProduct(MyApplication.getApplication().getDeviceAddress());
            if (connBleProduct != null) {
                connBleProduct.writeVibrate(i);
            }
        }
    }

    public void writeOpenClimax() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeOpenClimax();
        }
    }

    public void writeOpenClimax01() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeOpenClimax01();
        }
    }

    public void writeOpenPiu() {
        if (!ProductUtil.isFPiuType()) {
            Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
            while (it.hasNext()) {
                it.next().getValue().writeOpenPiu();
            }
        } else {
            BleProduct connBleProduct = getConnBleProduct(MyApplication.getApplication().getDeviceAddress());
            if (connBleProduct != null) {
                connBleProduct.writeOpenPiu();
            }
        }
    }

    public void writeOpenReversal() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeOpenReversal();
        }
    }

    public void writeOpenVoice() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeOpenVoice();
        }
    }

    public void writePosition(int i) {
        if (this.isWriteSpeed) {
            Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
            while (it.hasNext()) {
                it.next().getValue().writePosition(i);
            }
        }
    }

    public void writeRunAuto() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeRunAuto();
        }
    }

    public void writeRunManual() {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeRunManual();
        }
    }

    public void writeShutdown() {
        for (Map.Entry<String, BleProduct> entry : this.bleProducts.entrySet()) {
            entry.getValue().writeShutdown();
            entry.getValue().clearData();
        }
    }

    public void writeSmooth(int i) {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeSmooth(i);
        }
    }

    public void writeSpeed(int i) {
        BleProduct connBleProduct;
        if (!this.isWriteSpeed || (connBleProduct = getConnBleProduct(MyApplication.getApplication().getDeviceAddress())) == null) {
            return;
        }
        connBleProduct.writeSpeed(i);
    }

    public void writeSpeedMode(int i) {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeSpeedMode(i);
        }
    }

    public void writeStartPosition(int i) {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeStartPosition(i);
        }
    }

    public void writeStopPosition(int i) {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeStopPosition(i);
        }
    }

    public void writeVibrate(int i) {
        if (this.isWriteSpeed) {
            LogUtil.e("发送震动：" + i);
            BleProduct connBleProduct = getConnBleProduct(MyApplication.getApplication().getDeviceAddress());
            if (connBleProduct != null) {
                connBleProduct.writeVibrate(i);
            }
        }
    }

    public void writeVibrateMode(int i) {
        Iterator<Map.Entry<String, BleProduct>> it = this.bleProducts.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().writeVibrateMode(i);
        }
    }
}
