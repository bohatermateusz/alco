package com.hismith.hismithapp.utils;

import android.content.Context;

/* loaded from: classes3.dex */
public class ColorUtil {
    public static String toHexEncoding(Context context, int i) {
        StringBuffer stringBuffer = new StringBuffer("#");
        int color = context.getResources().getColor(i);
        int i2 = (16711680 & color) >> 16;
        int i3 = (65280 & color) >> 8;
        int i4 = color & 255;
        String hexString = Integer.toHexString(i2);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        stringBuffer.append(hexString);
        String hexString2 = Integer.toHexString(i3);
        if (hexString2.length() < 2) {
            hexString2 = "0" + hexString2;
        }
        stringBuffer.append(hexString2);
        String hexString3 = Integer.toHexString(i4);
        if (hexString3.length() < 2) {
            hexString3 = "0" + hexString3;
        }
        stringBuffer.append(hexString3);
        LogUtil.d("changecolor=" + stringBuffer.toString());
        return stringBuffer.toString();
    }
}
