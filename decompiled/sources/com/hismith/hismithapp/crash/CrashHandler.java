package com.hismith.hismithapp.crash;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Looper;
import android.os.Process;
import android.text.TextUtils;
import com.bumptech.glide.load.Key;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.Thread;
import java.lang.reflect.Field;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.HashMap;
import java.util.Map;

/* loaded from: classes3.dex */
public class CrashHandler implements Thread.UncaughtExceptionHandler {
    private static CrashHandler INSTANCE = new CrashHandler();
    private Context mContext;
    private Thread.UncaughtExceptionHandler mDefaultHandler;
    private Map<String, String> infos = new HashMap();
    private DateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH-mm-ss SSS");
    private String mFileName = "crashHandler.txt";

    private CrashHandler() {
    }

    public static CrashHandler getInstance() {
        return INSTANCE;
    }

    /* JADX WARN: Type inference failed for: r1v0, types: [com.hismith.hismithapp.crash.CrashHandler$1] */
    private boolean handleException(Throwable th) {
        if (th == null) {
            return false;
        }
        new Thread() { // from class: com.hismith.hismithapp.crash.CrashHandler.1
            @Override // java.lang.Thread, java.lang.Runnable
            public void run() {
                Looper.prepare();
                Looper.loop();
            }
        }.start();
        LogUtil.e("handleException: " + th.getMessage());
        collectDeviceInfo(this.mContext);
        saveCrashInfo2File(th);
        return false;
    }

    private void saveCrashInfo2File(Throwable th) {
        StringBuffer stringBuffer = new StringBuffer();
        stringBuffer.append(AppUtil.getAppName(this.mContext) + "_V" + AppUtil.getAppVersionName(this.mContext) + "\n");
        String str = (String) SharedPreferencesUtil.get(MyApplication.getApplication(), Constans.Token);
        if (!TextUtils.isEmpty(str)) {
            stringBuffer.append("token:" + str + "\n");
        }
        for (Map.Entry<String, String> entry : this.infos.entrySet()) {
            stringBuffer.append(entry.getKey() + "=" + entry.getValue() + "\n");
        }
        StringWriter stringWriter = new StringWriter();
        PrintWriter printWriter = new PrintWriter(stringWriter);
        th.printStackTrace(printWriter);
        for (Throwable cause = th.getCause(); cause != null; cause = cause.getCause()) {
            cause.printStackTrace(printWriter);
        }
        printWriter.close();
        stringBuffer.append(stringWriter.toString());
        try {
            String str2 = this.mFileName;
            File file = new File(this.mContext.getCacheDir().getPath() + "/crash_collection/");
            if (!file.exists()) {
                file.mkdirs();
            }
            File file2 = new File(file, str2);
            if (file2.exists()) {
                file2.delete();
                file2.createNewFile();
            } else {
                file2.createNewFile();
            }
            FileOutputStream fileOutputStream = new FileOutputStream(file2);
            fileOutputStream.write(stringBuffer.toString().getBytes(Key.STRING_CHARSET_NAME));
            fileOutputStream.close();
        } catch (Exception e) {
            LogUtil.e("an error occured while writing file...", e);
        }
    }

    public void collectDeviceInfo(Context context) {
        try {
            PackageInfo packageInfo = context.getPackageManager().getPackageInfo(context.getPackageName(), 1);
            if (packageInfo != null) {
                String str = packageInfo.versionName == null ? "null" : packageInfo.versionName;
                String str2 = packageInfo.versionCode + "";
                this.infos.put("versionName", str);
                this.infos.put("versionCode", str2);
            }
        } catch (PackageManager.NameNotFoundException e) {
            LogUtil.e("an error occured when collect package info", e);
        }
        for (Field field : Build.class.getDeclaredFields()) {
            try {
                field.setAccessible(true);
                this.infos.put(field.getName(), field.get(null).toString());
                LogUtil.d(field.getName() + " : " + field.get(null));
            } catch (Exception e2) {
                LogUtil.e("an error occured when collect crash info", e2);
            }
        }
    }

    public void init(Context context) {
        this.mContext = context;
        this.mDefaultHandler = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler(this);
    }

    @Override // java.lang.Thread.UncaughtExceptionHandler
    public void uncaughtException(Thread thread, Throwable th) {
        Thread.UncaughtExceptionHandler uncaughtExceptionHandler;
        if (!handleException(th) && (uncaughtExceptionHandler = this.mDefaultHandler) != null) {
            uncaughtExceptionHandler.uncaughtException(thread, th);
            return;
        }
        try {
            Thread.sleep(500L);
        } catch (InterruptedException e) {
            LogUtil.e("error : ", e);
        }
        Process.killProcess(Process.myPid());
        System.exit(1);
    }
}
