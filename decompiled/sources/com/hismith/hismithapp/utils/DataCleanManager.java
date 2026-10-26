package com.hismith.hismithapp.utils;

import android.content.Context;
import android.os.Environment;
import android.os.Looper;
import androidx.exifinterface.media.ExifInterface;
import com.bumptech.glide.Glide;
import java.io.File;
import java.math.BigDecimal;

/* loaded from: classes3.dex */
public class DataCleanManager {
    public static void cleanApplicationData(Context context, String... strArr) {
        cleanImageMemoryCache(context);
        cleanImageDiskCache(context);
        cleanInternalCache(context);
        cleanExternalCache(context);
        for (String str : strArr) {
            cleanCustomCache(str);
        }
    }

    public static void cleanCustomCache(String str) {
        deleteFilesByDirectory(new File(str));
    }

    public static void cleanDatabaseByName(Context context, String str) {
        context.deleteDatabase(str);
    }

    public static void cleanDatabases(Context context) {
        deleteFilesByDirectory(new File("/data/data/" + context.getPackageName() + "/databases"));
    }

    public static void cleanExternalCache(Context context) {
        if (Environment.getExternalStorageState().equals("mounted")) {
            deleteFilesByDirectory(context.getExternalCacheDir());
        }
    }

    public static void cleanFiles(Context context) {
        deleteFilesByDirectory(context.getFilesDir());
    }

    private static void cleanImageDiskCache(final Context context) {
        try {
            if (Looper.myLooper() == Looper.getMainLooper()) {
                new Thread(new Runnable() { // from class: com.hismith.hismithapp.utils.DataCleanManager.1
                    @Override // java.lang.Runnable
                    public void run() {
                        Glide.get(context).clearDiskCache();
                    }
                }).start();
            } else {
                Glide.get(context).clearDiskCache();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void cleanImageMemoryCache(Context context) {
        try {
            if (Looper.myLooper() == Looper.getMainLooper()) {
                Glide.get(context).clearMemory();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void cleanInternalCache(Context context) {
        deleteFilesByDirectory(context.getCacheDir());
    }

    public static void cleanSharedPreference(Context context) {
        deleteFilesByDirectory(new File("/data/data/" + context.getPackageName() + "/shared_prefs"));
    }

    private static void deleteFilesByDirectory(File file) {
        if (file != null) {
            try {
                if (file.exists() && file.isDirectory()) {
                    for (File file2 : file.listFiles()) {
                        if (file2.isDirectory()) {
                            deleteFilesByDirectory(file2);
                        } else {
                            file2.delete();
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static String getCacheSize(Context context) {
        try {
            long folderSize = getFolderSize(context.getCacheDir());
            if (Environment.getExternalStorageState().equals("mounted")) {
                folderSize += getFolderSize(context.getExternalCacheDir());
            }
            return getFormatSize(folderSize).toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    public static long getFolderSize(File file) throws Exception {
        long j = 0;
        try {
            File[] listFiles = file.listFiles();
            for (int i = 0; i < listFiles.length; i++) {
                j += listFiles[i].isDirectory() ? getFolderSize(listFiles[i]) : listFiles[i].length();
            }
            return j;
        } catch (Exception e) {
            e.printStackTrace();
            return j;
        }
    }

    public static String getFormatSize(double d) throws Exception {
        double d2 = d / 1024.0d;
        if (d2 < 1.0d) {
            return d + "B";
        }
        double d3 = d2 / 1024.0d;
        if (d3 < 1.0d) {
            return new BigDecimal(Double.toString(d2)).setScale(2, 4).toPlainString() + "K";
        }
        double d4 = d3 / 1024.0d;
        if (d4 < 1.0d) {
            return new BigDecimal(Double.toString(d3)).setScale(2, 4).toPlainString() + "M";
        }
        double d5 = d4 / 1024.0d;
        if (d5 < 1.0d) {
            return new BigDecimal(Double.toString(d4)).setScale(2, 4).toPlainString() + "G";
        }
        return new BigDecimal(d5).setScale(2, 4).toPlainString() + ExifInterface.GPS_DIRECTION_TRUE;
    }
}
