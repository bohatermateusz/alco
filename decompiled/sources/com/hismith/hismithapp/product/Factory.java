package com.hismith.hismithapp.product;

import android.bluetooth.BluetoothGatt;
import android.content.Context;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.utils.Constans;

/* loaded from: classes3.dex */
public class Factory {
    public static String buildDeviceType(ProductModel.Fun[] funArr) {
        if (funArr == null || funArr.length <= 0) {
            return Constans.DEVICE_TYPE_SM;
        }
        String str = funArr[0].getfType();
        for (int i = 1; i < funArr.length; i++) {
            if (!funArr[i].getfType().equals("S")) {
                if (funArr[i].getfType().equals("M")) {
                    str = Constans.DEVICE_TYPE_SM2;
                } else if (funArr[i].getfType().equals(Constans.DEVICE_TYPE_SM)) {
                    str = Constans.DEVICE_TYPE_SMV;
                } else if (funArr[i].getfType().equals(Constans.DEVICE_TYPE_SMS)) {
                    str = Constans.DEVICE_TYPE_SMS;
                }
            }
        }
        return str;
    }

    public static ProductBaseControl createProduct(Context context, BluetoothGatt bluetoothGatt, ProductModel productModel) {
        return productModel.getCommand().equals("CC") ? new ProductCCControl(context, bluetoothGatt, productModel) : new ProductAAControl(context, bluetoothGatt, productModel);
    }
}
