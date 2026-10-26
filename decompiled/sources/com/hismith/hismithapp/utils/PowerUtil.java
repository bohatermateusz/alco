package com.hismith.hismithapp.utils;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.PowerManager;
import android.util.Log;

/* loaded from: classes.dex */
public class PowerUtil {
    public static boolean isIgnoringBatteryOptimizations(Context context) {
        if (((PowerManager) context.getSystemService("power")).isIgnoringBatteryOptimizations(context.getPackageName())) {
            Log.e("BatteryOptimization", "App is ignoring battery optimizations.");
            return true;
        }
        Log.e("BatteryOptimization", "App is NOT ignoring battery optimizations.");
        return false;
    }

    public static void requestBatteryOptimizations(Context context) {
        String packageName = context.getPackageName();
        Intent intent = new Intent("android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS");
        intent.setData(Uri.parse("package:" + packageName));
        context.startActivity(intent);
    }
}
