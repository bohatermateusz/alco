package com.hismith.hismithapp.utils;

import android.content.Context;
import android.content.Intent;
import android.location.LocationManager;
import android.provider.Settings;
import com.google.firebase.analytics.FirebaseAnalytics;

/* loaded from: classes.dex */
public class GPSUtil {
    public static boolean isOpen(Context context) {
        return Settings.Secure.getInt(context.getContentResolver(), "location_mode", 0) != 0;
    }

    public static final void openGPS(Context context) {
        LocationManager locationManager = (LocationManager) context.getSystemService(FirebaseAnalytics.Param.LOCATION);
        boolean isProviderEnabled = locationManager.isProviderEnabled("gps");
        boolean isProviderEnabled2 = locationManager.isProviderEnabled("network");
        if (isProviderEnabled || isProviderEnabled2) {
            return;
        }
        Intent intent = new Intent();
        intent.setAction("android.settings.LOCATION_SOURCE_SETTINGS");
        intent.addFlags(268435456);
        context.startActivity(intent);
    }

    public static void openGPS(Context context, boolean z) {
        if (z) {
            Settings.Secure.putInt(context.getContentResolver(), "location_mode", 2);
        } else {
            Settings.Secure.putInt(context.getContentResolver(), "location_mode", 0);
        }
    }
}
