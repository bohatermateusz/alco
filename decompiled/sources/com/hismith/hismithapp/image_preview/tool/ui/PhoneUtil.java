package com.hismith.hismithapp.image_preview.tool.ui;

import android.content.Context;
import android.util.DisplayMetrics;
import android.view.WindowManager;

/* loaded from: classes3.dex */
public class PhoneUtil {
    public static int dp2px(Context context, float f) {
        return (int) ((f * context.getApplicationContext().getResources().getDisplayMetrics().density) + 0.5f);
    }

    public static int getPhoneHei(Context context) {
        WindowManager windowManager = (WindowManager) context.getApplicationContext().getSystemService("window");
        DisplayMetrics displayMetrics = new DisplayMetrics();
        windowManager.getDefaultDisplay().getMetrics(displayMetrics);
        return displayMetrics.heightPixels;
    }

    public static float getPhoneRatio(Context context) {
        return getPhoneHei(context) / getPhoneWid(context);
    }

    public static int getPhoneWid(Context context) {
        WindowManager windowManager = (WindowManager) context.getApplicationContext().getSystemService("window");
        DisplayMetrics displayMetrics = new DisplayMetrics();
        windowManager.getDefaultDisplay().getMetrics(displayMetrics);
        return displayMetrics.widthPixels;
    }

    public static int px2dp(Context context, float f) {
        return (int) ((f / context.getApplicationContext().getResources().getDisplayMetrics().density) + 0.5f);
    }
}
