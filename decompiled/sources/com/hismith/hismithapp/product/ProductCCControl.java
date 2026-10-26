package com.hismith.hismithapp.product;

import android.bluetooth.BluetoothGatt;
import android.content.Context;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.ModeDataVCUtil;

/* loaded from: classes3.dex */
public class ProductCCControl extends ProductBaseControl {
    private Context mContext;

    public ProductCCControl(Context context, BluetoothGatt bluetoothGatt) {
        super(context, bluetoothGatt);
        this.mContext = context;
    }

    public ProductCCControl(Context context, BluetoothGatt bluetoothGatt, ProductModel productModel) {
        super(context, bluetoothGatt, productModel);
        this.mContext = context;
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeClosePiu() {
        writeCCData(Constans.piuCommand, "00");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeCloseReversal() {
        writeCCData(ModeDataVCUtil.HEAD_VERSION, "C0");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeCloseVibrate() {
        writeCCData(5, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeCloseVoice() {
        writeCCData(ModeDataVCUtil.HEAD_VERSION, "B0");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetBattery() {
        writeCCData(0, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetMode() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetSpeed() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetStatus() {
        writeCCData(1, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetVibrate() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeGetVoice() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenClimax() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenClimax01() {
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenPiu() {
        writeCCData(Constans.piuCommand, ModeDataVCUtil.HEAD_VERSION);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenReversal() {
        writeCCData(ModeDataVCUtil.HEAD_VERSION, "C1");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeOpenVoice() {
        writeCCData(ModeDataVCUtil.HEAD_VERSION, "B1");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writePosition(int i) {
        writeCCData(10, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeRunAuto() {
        writeCCData(ModeDataVCUtil.HEAD_VERSION, "A0");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeRunManual() {
        writeCCData(ModeDataVCUtil.HEAD_VERSION, "A1");
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeShutdown() {
        writeCCData(2, 0);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSmooth(int i) {
        writeCCData(9, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSpeed(int i) {
        writeCCData(3, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSpeedMode(int i) {
        writeCCData(4, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeSpeedMode(String str) {
        writeData(str);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeStartPosition(int i) {
        writeCCData(7, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeStopPosition(int i) {
        writeCCData(8, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeVibrate(int i) {
        writeCCData(5, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeVibrateMode(int i) {
        writeCCData(6, i);
    }

    @Override // com.hismith.hismithapp.product.ProductBaseControl
    public void writeVibrateMode(String str) {
        writeData(str);
    }
}
