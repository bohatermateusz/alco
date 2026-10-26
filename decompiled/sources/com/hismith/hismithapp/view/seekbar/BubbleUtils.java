package com.hismith.hismithapp.view.seekbar;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.text.TextUtils;
import android.util.TypedValue;
import java.util.Properties;

/* loaded from: classes3.dex */
class BubbleUtils {
    private static final String KEY_MIUI_MANE = "ro.miui.ui.version.name";
    private static Boolean miui;
    private static Properties sProperties = new Properties();

    BubbleUtils() {
    }

    static int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, Resources.getSystem().getDisplayMetrics());
    }

    static Bitmap imageScale(Bitmap bitmap, int i, int i2) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(i / width, i2 / height);
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
    }

    static boolean isMIUI() {
        Boolean bool;
        Boolean bool2 = miui;
        if (bool2 != null) {
            return bool2.booleanValue();
        }
        try {
            bool = Boolean.valueOf(!TextUtils.isEmpty((String) Class.forName("android.os.SystemProperties").getDeclaredMethod("get", String.class).invoke(null, KEY_MIUI_MANE)));
            miui = bool;
        } catch (Exception unused) {
            bool = false;
            miui = bool;
        }
        return bool.booleanValue();
    }

    static int sp2px(int i) {
        return (int) TypedValue.applyDimension(2, i, Resources.getSystem().getDisplayMetrics());
    }
}
