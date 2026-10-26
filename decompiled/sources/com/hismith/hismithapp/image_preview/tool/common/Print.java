package com.hismith.hismithapp.image_preview.tool.common;

import android.text.TextUtils;
import android.util.Log;

/* loaded from: classes3.dex */
public class Print {
    public static void d(String str, String str2) {
        if (TextUtils.isEmpty(str)) {
            str = "shinichi.cc";
        }
        if (str2 == null || "".equals(str2)) {
            Log.e(str, "msg == null");
            return;
        }
        int length = str2.length();
        int i = 0;
        int i2 = 2000;
        int i3 = 0;
        while (i < 100) {
            if (length <= i2) {
                Log.d(str, str2.substring(i3, length));
                return;
            }
            Log.d(str + i, str2.substring(i3, i2));
            i++;
            i3 = i2;
            i2 += 2000;
        }
    }
}
