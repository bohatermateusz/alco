package com.hismith.hismithapp.product;

import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.content.Context;
import android.text.TextUtils;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HexUtil;
import com.hismith.hismithapp.utils.LogUtil;
import io.reactivex.Observable;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.functions.Consumer;
import io.reactivex.schedulers.Schedulers;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;

/* loaded from: classes3.dex */
public abstract class ProductBaseControl {
    private static long SEND_INTERVAL_MS = 100;
    private final ConcurrentHashMap<Integer, String> commandMap;
    private final CompositeDisposable disposables;
    private final Object lock;
    protected BluetoothGatt mBluetoothGatt;
    private Context mContext;
    protected ProductModel mProductModel;
    private final ConcurrentLinkedQueue<Integer> orderQueue;

    public ProductBaseControl(Context context, BluetoothGatt bluetoothGatt) {
        this(context, bluetoothGatt, null);
    }

    public ProductBaseControl(Context context, BluetoothGatt bluetoothGatt, ProductModel productModel) {
        this.disposables = new CompositeDisposable();
        this.orderQueue = new ConcurrentLinkedQueue<>();
        this.commandMap = new ConcurrentHashMap<>();
        this.lock = new Object();
        this.mContext = context;
        this.mProductModel = productModel;
        this.mBluetoothGatt = bluetoothGatt;
        if (productModel != null && productModel.getCommand().equals("CC")) {
            SEND_INTERVAL_MS = 50L;
        }
        setupProcessingPipeline();
    }

    private void addPendingData(int i, String str) {
        LogUtil.e("addPendingData data0:" + i + ", hex:" + str);
        if (TextUtils.isEmpty(str)) {
            return;
        }
        synchronized (this.lock) {
            if (!this.commandMap.containsKey(Integer.valueOf(i))) {
                this.orderQueue.offer(Integer.valueOf(i));
            }
            this.commandMap.put(Integer.valueOf(i), str);
        }
    }

    private void sendCommand(String str) {
        BluetoothGatt bluetoothGatt;
        BluetoothGattService service;
        BluetoothGattCharacteristic characteristic;
        if (TextUtils.isEmpty(str) || (bluetoothGatt = this.mBluetoothGatt) == null || (service = bluetoothGatt.getService(Constans.UUID_BLUETOOTH_DATA)) == null || (characteristic = service.getCharacteristic(Constans.UUID_BLUETOOTH_DATA_WRITH)) == null) {
            return;
        }
        byte[] hexStringToBytes = HexUtil.hexStringToBytes(str);
        if (hexStringToBytes.length <= 20) {
            characteristic.setValue(hexStringToBytes);
            LogUtil.e(this.mBluetoothGatt.getDevice().getName() + ", 发送指令：" + str + ", 成功：" + this.mBluetoothGatt.writeCharacteristic(characteristic));
            return;
        }
        LogUtil.e("sendCommand: 分包 length=" + hexStringToBytes.length);
        int length = (hexStringToBytes.length + 19) / 20;
        for (int i = 0; i < length; i++) {
            int i2 = i * 20;
            int min = Math.min(20, hexStringToBytes.length - i2);
            byte[] bArr = new byte[min];
            System.arraycopy(hexStringToBytes, i2, bArr, 0, min);
            characteristic.setValue(bArr);
            this.mBluetoothGatt.writeCharacteristic(characteristic);
        }
    }

    private void sendPendingData() {
        String remove;
        synchronized (this.lock) {
            if (this.commandMap.containsKey(2)) {
                remove = this.commandMap.remove(2);
                this.orderQueue.clear();
                this.commandMap.clear();
            } else {
                Integer poll = this.orderQueue.poll();
                remove = poll != null ? this.commandMap.remove(poll) : null;
            }
        }
        if (remove != null) {
            sendCommand(remove);
        }
    }

    private void setupProcessingPipeline() {
        long j = SEND_INTERVAL_MS;
        this.disposables.add(Observable.interval(j, j, TimeUnit.MILLISECONDS).observeOn(Schedulers.io()).subscribe(new Consumer() { // from class: com.hismith.hismithapp.product.ProductBaseControl$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                ProductBaseControl.this.m620xa2e9706f((Long) obj);
            }
        }, new Consumer() { // from class: com.hismith.hismithapp.product.ProductBaseControl$$ExternalSyntheticLambda1
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                LogUtil.e("发送定时器错误", (Throwable) obj);
            }
        }));
    }

    public void dispose() {
        if (this.disposables.isDisposed()) {
            return;
        }
        this.disposables.dispose();
        synchronized (this.lock) {
            this.orderQueue.clear();
            this.commandMap.clear();
        }
        LogUtil.d("资源已清理");
    }

    public boolean isDisposed() {
        return this.disposables.isDisposed();
    }

    /* renamed from: lambda$setupProcessingPipeline$0$com-hismith-hismithapp-product-ProductBaseControl, reason: not valid java name */
    /* synthetic */ void m620xa2e9706f(Long l) throws Exception {
        if (this.commandMap.isEmpty()) {
            return;
        }
        sendPendingData();
    }

    protected void writeCCData(int i, int i2) {
        String hexString = Integer.toHexString(i);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        String hexString2 = Integer.toHexString(i2);
        if (hexString2.length() < 2) {
            hexString2 = "0" + hexString2;
        }
        String hexString3 = Integer.toHexString(i2 + i);
        if (hexString3.length() < 2) {
            hexString3 = "0" + hexString3;
        }
        addPendingData(i, "CC" + hexString + hexString2 + hexString3);
    }

    protected void writeCCData(String str, String str2) {
        int intValue = Integer.valueOf(str, 16).intValue();
        String hexString = Integer.toHexString(Integer.valueOf(str2, 16).intValue() + intValue);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        addPendingData(intValue, "CC" + str + str2 + hexString);
    }

    public abstract void writeClosePiu();

    public abstract void writeCloseReversal();

    public abstract void writeCloseVibrate();

    public abstract void writeCloseVoice();

    protected void writeData(int i, int i2) {
        String hexString = Integer.toHexString(i);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        String hexString2 = Integer.toHexString(i2);
        if (hexString2.length() < 2) {
            hexString2 = "0" + hexString2;
        }
        String hexString3 = Integer.toHexString(i2 + i);
        if (hexString3.length() < 2) {
            hexString3 = "0" + hexString3;
        }
        addPendingData(i, "AA" + hexString + hexString2 + hexString3);
    }

    public void writeData(String str) {
        int i = -1;
        if (str != null && str.length() == 8) {
            try {
                i = Integer.parseInt(str.substring(2, 4), 16);
            } catch (NumberFormatException unused) {
            }
        }
        addPendingData(i, str);
    }

    protected void writeData(String str, String str2) {
        int intValue = Integer.valueOf(str, 16).intValue();
        String hexString = Integer.toHexString(Integer.valueOf(str2, 16).intValue() + intValue);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        addPendingData(intValue, "AA" + str + str2 + hexString);
    }

    public abstract void writeGetBattery();

    public abstract void writeGetMode();

    public abstract void writeGetSpeed();

    public abstract void writeGetStatus();

    public abstract void writeGetVibrate();

    public abstract void writeGetVoice();

    public abstract void writeOpenClimax();

    public abstract void writeOpenClimax01();

    public abstract void writeOpenPiu();

    public abstract void writeOpenReversal();

    public abstract void writeOpenVoice();

    public abstract void writePosition(int i);

    public abstract void writeRunAuto();

    public abstract void writeRunManual();

    public abstract void writeShutdown();

    public abstract void writeSmooth(int i);

    public abstract void writeSpeed(int i);

    public abstract void writeSpeedMode(int i);

    public abstract void writeSpeedMode(String str);

    public abstract void writeStartPosition(int i);

    public abstract void writeStopPosition(int i);

    public abstract void writeVibrate(int i);

    public abstract void writeVibrateMode(int i);

    public abstract void writeVibrateMode(String str);
}
