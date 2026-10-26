package com.hismith.hismithapp.utils;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;

/* loaded from: classes3.dex */
public class InstallApkUtil {
    public static boolean checkInstallPackage(Context context) {
        return context.getPackageManager().canRequestPackageInstalls();
    }

    public static void installUnknownApp(Context context) {
    }

    public static void startUnknownAppSources(Activity activity) {
        activity.startActivityForResult(new Intent("android.settings.MANAGE_UNKNOWN_APP_SOURCES"), 19);
    }
}
