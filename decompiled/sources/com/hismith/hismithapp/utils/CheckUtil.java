package com.hismith.hismithapp.utils;

import android.app.Activity;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Binder;
import android.util.DisplayMetrics;
import java.lang.reflect.InvocationTargetException;

/* loaded from: classes3.dex */
public class CheckUtil {
    public static int checkOp(Context context, int i) {
        Object systemService = context.getSystemService("appops");
        try {
            return ((Integer) systemService.getClass().getDeclaredMethod("checkOp", Integer.TYPE, Integer.TYPE, String.class).invoke(systemService, Integer.valueOf(i), Integer.valueOf(Binder.getCallingUid()), context.getPackageName())).intValue();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            return -1;
        } catch (IllegalArgumentException e2) {
            e2.printStackTrace();
            return -1;
        } catch (NoSuchMethodException e3) {
            e3.printStackTrace();
            return -1;
        } catch (InvocationTargetException e4) {
            e4.printStackTrace();
            return -1;
        }
    }

    public static int getFloatPermissionStatus(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("context is null");
        }
        Cursor query = context.getContentResolver().query(Uri.parse("content://com.iqoo.secure.provider.secureprovider/allowfloatwindowapp"), null, "pkgname = ?", new String[]{context.getPackageName()}, null);
        if (query == null) {
            return getFloatPermissionStatus2(context);
        }
        query.getColumnNames();
        if (!query.moveToFirst()) {
            query.close();
            return getFloatPermissionStatus2(context);
        }
        int i = query.getInt(query.getColumnIndex("currentlmode"));
        query.close();
        return i;
    }

    private static int getFloatPermissionStatus2(Context context) {
        Cursor query = context.getContentResolver().query(Uri.parse("content://com.vivo.permissionmanager.provider.permission/float_window_apps"), null, "pkgname = ?", new String[]{context.getPackageName()}, null);
        if (query != null) {
            if (query.moveToFirst()) {
                int i = query.getInt(query.getColumnIndex("currentmode"));
                query.close();
                return i;
            }
            query.close();
        }
        return 1;
    }

    public static int[] pixel(Activity activity) {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        activity.getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        return new int[]{displayMetrics.widthPixels, displayMetrics.heightPixels};
    }
}
