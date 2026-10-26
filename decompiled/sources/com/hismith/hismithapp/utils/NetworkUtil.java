package com.hismith.hismithapp.utils;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.widget.Toast;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class NetworkUtil {
    public static boolean isConnectedByWifi(Context context) {
        NetworkInfo activeNetworkInfo = ((ConnectivityManager) context.getSystemService("connectivity")).getActiveNetworkInfo();
        return activeNetworkInfo != null && activeNetworkInfo.getType() == 1;
    }

    public static boolean isNetDisconnected(Context context) {
        if (isNetworkConnected(context)) {
            return false;
        }
        Toast.makeText(context, R.string.android_networkConnectionTimedOut, 0).show();
        return true;
    }

    public static boolean isNetworkConnected(Context context) {
        NetworkInfo activeNetworkInfo;
        if (context == null || (activeNetworkInfo = ((ConnectivityManager) context.getSystemService("connectivity")).getActiveNetworkInfo()) == null) {
            return false;
        }
        return activeNetworkInfo.isAvailable();
    }
}
