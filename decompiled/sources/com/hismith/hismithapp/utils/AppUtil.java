package com.hismith.hismithapp.utils;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.Uri;
import android.os.Build;
import kotlin.time.DurationKt;

/* loaded from: classes3.dex */
public class AppUtil {
    public static synchronized String getAppName(Context context) {
        String str;
        synchronized (AppUtil.class) {
            str = "";
            try {
                str = context.getResources().getString(context.getPackageManager().getPackageInfo(context.getPackageName(), 0).applicationInfo.labelRes);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return str;
    }

    public static int getAppVersionCode(Context context) {
        try {
            return context.getPackageManager().getPackageInfo(context.getPackageName(), 0).versionCode;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    public static String getAppVersionName(Context context) {
        try {
            return context.getPackageManager().getPackageInfo(context.getPackageName(), 0).versionName;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static String getPackageName(Context context) {
        try {
            return context.getPackageManager().getPackageInfo(context.getPackageName(), 0).packageName;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static boolean isDeviceExpired(Context context) {
        boolean booleanValue = ((Boolean) SharedPreferencesUtil.get(context, Constans.COMMUNITY_HIDE, false)).booleanValue();
        long longValue = ((Long) SharedPreferencesUtil.get(context, Constans.LAST_CONNECTED_DEVICE_TIME, 0L)).longValue();
        if (longValue == 0) {
            return true;
        }
        long currentTimeMillis = (System.currentTimeMillis() - longValue) / DurationKt.MILLIS_IN_DAY;
        LogUtil.e("isDeviceExpired :" + currentTimeMillis);
        return booleanValue || currentTimeMillis >= 14;
    }

    public static boolean isDownloadManagerAvailable(Context context) {
        int applicationEnabledSetting = context.getPackageManager().getApplicationEnabledSetting("com.android.providers.downloads");
        return (applicationEnabledSetting == 2 || applicationEnabledSetting == 3 || applicationEnabledSetting == 4) ? false : true;
    }

    public static void registerReceiver(Context context, BroadcastReceiver broadcastReceiver, IntentFilter intentFilter) {
        if (Build.VERSION.SDK_INT >= 34) {
            context.registerReceiver(broadcastReceiver, intentFilter, 2);
        } else {
            context.registerReceiver(broadcastReceiver, intentFilter);
        }
    }

    public static void shareAppDl(Context context, String str) {
        Intent intent = new Intent("android.intent.action.VIEW", Uri.parse("market://details?id=" + str));
        intent.addFlags(268435456);
        context.startActivity(intent);
    }

    public static void startUriToWeb(Context context, String str) {
        try {
            if (!str.matches("^[a-zA-Z]+://.*")) {
                if (str.startsWith("//")) {
                    str = "https:" + str;
                } else {
                    if (!str.contains(".") || str.contains(" ")) {
                        throw new IllegalArgumentException("无效的URL格式");
                    }
                    str = "https://" + str;
                }
            }
            LogUtil.e("****fixedUrl:" + str);
            Intent intent = new Intent("android.intent.action.VIEW", Uri.parse(str));
            intent.addFlags(268435456);
            if (intent.resolveActivity(context.getPackageManager()) != null) {
                context.startActivity(intent);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
