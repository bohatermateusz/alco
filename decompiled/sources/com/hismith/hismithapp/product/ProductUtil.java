package com.hismith.hismithapp.product;

import android.content.Context;
import android.text.TextUtils;
import com.google.gson.Gson;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.manager.ProductManager;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.ProductsModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;

/* loaded from: classes3.dex */
public class ProductUtil {
    public static ProductModel.Fun getProductFun(int i) {
        return MyApplication.getApplication().getmProductModel().getFuns()[i];
    }

    public static ProductModel.Fun getProductFun(ProductModel productModel, int i) {
        return productModel.getFuns()[i];
    }

    public static ProductModel.Mode[] getProductMode(int i) {
        return MyApplication.getApplication().getmProductModel().getFuns()[i].getModes();
    }

    public static ProductModel.Mode[] getProductMode(ProductModel productModel, int i) {
        return productModel.getFuns()[i].getModes();
    }

    public static ProductModel getProductModel(Context context, String str) {
        BleProduct connBleProduct = BleManager.getInstance(context).getConnBleProduct(str);
        if (connBleProduct == null) {
            return null;
        }
        return connBleProduct.getmProductModel();
    }

    public static ProductModel getProductModelForCode(Context context, String str) {
        try {
            String str2 = (String) SharedPreferencesUtil.get(context, Constans.ProductAllData, "");
            if (TextUtils.isEmpty(str2)) {
                return ProductManager.getProduct(str);
            }
            ProductsModel productsModel = (ProductsModel) new Gson().fromJson(str2, ProductsModel.class);
            if (productsModel == null || productsModel.getProducts() == null || productsModel.getProducts().length <= 0) {
                return null;
            }
            for (int i = 0; i < productsModel.getProducts().length; i++) {
                if (productsModel.getProducts()[i].getCode().equals(str)) {
                    return productsModel.getProducts()[i];
                }
            }
            return null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static String getProductType(Context context, String str) {
        ProductModel productModel;
        BleProduct connBleProduct = BleManager.getInstance(context).getConnBleProduct(str);
        return (connBleProduct == null || (productModel = connBleProduct.getmProductModel()) == null) ? "" : productModel.getDeviceType();
    }

    public static boolean hasFun(int i) {
        return (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getFuns() == null || MyApplication.getApplication().getmProductModel().getFuns().length <= i) ? false : true;
    }

    public static boolean hasFun(ProductModel productModel, int i) {
        return (productModel == null || productModel.getFuns() == null || productModel.getFuns().length <= i) ? false : true;
    }

    public static boolean hasMode(int i) {
        return hasMode(i, 0);
    }

    public static boolean hasMode(int i, int i2) {
        return MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getFuns() != null && MyApplication.getApplication().getmProductModel().getFuns().length > i && MyApplication.getApplication().getmProductModel().getFuns()[i].getModes() != null && MyApplication.getApplication().getmProductModel().getFuns()[i].getModes().length > 0 && MyApplication.getApplication().getmProductModel().getFuns()[i].getModes().length >= i2;
    }

    public static boolean hasMode(ProductModel productModel, int i) {
        return hasMode(productModel, i, 0);
    }

    public static boolean hasMode(ProductModel productModel, int i, int i2) {
        return (productModel == null || productModel.getFuns() == null || productModel.getFuns().length <= i || productModel.getFuns()[i].getModes() == null || productModel.getFuns()[i].getModes().length < i2) ? false : true;
    }

    public static boolean hasReversal() {
        return MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getReversal() == 1;
    }

    public static boolean hasSFun(int i) {
        return MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getFuns() != null && MyApplication.getApplication().getmProductModel().getFuns().length > i && MyApplication.getApplication().getmProductModel().getFuns()[i].getfType().toUpperCase().contains("S");
    }

    public static boolean hasSFun(ProductModel productModel, int i) {
        return productModel != null && productModel.getFuns() != null && productModel.getFuns().length > i && productModel.getFuns()[i].getfType().toUpperCase().contains("S");
    }

    public static boolean hasSType() {
        return MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getDeviceType().contains("S");
    }

    public static boolean isFPiuType() {
        if (MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_FUN.equals(MyApplication.getApplication().getmProductModel().getDeviceType()) && MyApplication.getApplication().getmProductModel().getFuns() != null) {
            for (int i = 0; i < MyApplication.getApplication().getmProductModel().getFuns().length; i++) {
                if (Constans.piuCommand.equals(MyApplication.getApplication().getmProductModel().getFuns()[i].getfCommand())) {
                    return true;
                }
            }
        }
        return false;
    }

    public static boolean isFType() {
        return MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_FUN.equals(MyApplication.getApplication().getmProductModel().getDeviceType());
    }

    public static boolean isFType(Context context, String str) {
        String productType = getProductType(context, str);
        return !TextUtils.isEmpty(productType) && Constans.DEVICE_TYPE_FUN.equals(productType);
    }

    public static boolean isSM2Type() {
        return MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_SM2.equals(MyApplication.getApplication().getmProductModel().getDeviceType());
    }

    public static boolean isSM2Type(Context context, String str) {
        String productType = getProductType(context, str);
        return !TextUtils.isEmpty(productType) && Constans.DEVICE_TYPE_SM2.equals(productType);
    }

    public static boolean isSMSType() {
        return MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType());
    }

    public static boolean isSMSType(Context context, String str) {
        String productType = getProductType(context, str);
        return !TextUtils.isEmpty(productType) && Constans.DEVICE_TYPE_SMS.equals(productType);
    }

    public static boolean isSMType() {
        return MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_SM.equals(MyApplication.getApplication().getmProductModel().getDeviceType());
    }

    public static boolean isSMType(Context context, String str) {
        String productType = getProductType(context, str);
        return !TextUtils.isEmpty(productType) && Constans.DEVICE_TYPE_SM.equals(productType);
    }

    public static boolean isSMVType() {
        return MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_SMV.equals(MyApplication.getApplication().getmProductModel().getDeviceType());
    }

    public static boolean isSMVType(Context context, String str) {
        String productType = getProductType(context, str);
        return !TextUtils.isEmpty(productType) && Constans.DEVICE_TYPE_SMV.equals(productType);
    }
}
