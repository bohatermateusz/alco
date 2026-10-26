package com.hismith.hismithapp.product;

import android.bluetooth.BluetoothGatt;
import android.content.Context;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.utils.ModeDataVCUtil;

/* loaded from: classes3.dex */
public class ProductAAControl extends ProductBaseControl {
    private Context mContext;

    public ProductAAControl(Context context, BluetoothGatt bluetoothGatt) {
        super(context, bluetoothGatt);
        this.mContext = context;
    }

    public ProductAAControl(Context context, BluetoothGatt bluetoothGatt, ProductModel productModel) {
        super(context, bluetoothGatt, productModel);
        this.mContext = context;
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeClosePiu() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeCloseReversal() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeCloseVibrate() {
        writeData("06", "F0");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeCloseVoice() {
        writeData("07", "F0");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetBattery() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetMode() {
        writeData(5, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetSpeed() {
        writeData(3, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetStatus() {
        writeData(1, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetVibrate() {
        writeData(6, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetVoice() {
        writeData(7, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenClimax() {
        writeData(6, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenClimax01() {
        writeData(1, 161);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenPiu() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenReversal() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenVoice() {
        writeData("07", ModeDataVCUtil.HEAD_VERSION);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writePosition(int i) {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeRunAuto() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeRunManual() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeShutdown() {
        writeData(2, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSmooth(int i) {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSpeed(int i) {
        writeData(4, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSpeedMode(int i) {
        writeData(5, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSpeedMode(String str) {
        writeData(str);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeStartPosition(int i) {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeStopPosition(int i) {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeVibrate(int i) {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeVibrateMode(int i) {
        writeData(6, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeVibrateMode(String str) {
        writeData(str);
    }
}
