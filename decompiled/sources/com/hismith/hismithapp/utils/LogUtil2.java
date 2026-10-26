package com.hismith.hismithapp.utils;

import android.util.Log;

/* loaded from: classes3.dex */
public class LogUtil2 {
    static String className;
    static int lineNumber;
    private static Callback mCallback;
    static String methodName;

    public interface Callback {
        void onCallback(String str);
    }

    private static String createLog(String str) {
        StringBuffer stringBuffer = new StringBuffer("================");
        stringBuffer.append(methodName);
        stringBuffer.append("(").append(className).append(":").append(lineNumber).append(")================:");
        stringBuffer.append(str);
        Callback callback = mCallback;
        if (callback != null) {
            callback.onCallback(stringBuffer.toString());
        }
        return stringBuffer.toString();
    }

    public static void d(String str) {
        if (isDebuggable()) {
            getMethodNames(new Throwable().getStackTrace());
            Log.d(className, createLog(str));
        }
    }

    public static void e(String str) {
        if (isDebuggable()) {
            getMethodNames(new Throwable().getStackTrace());
            Log.e(className, createLog(str));
        }
    }

    public static void e(String str, Throwable th) {
        if (isDebuggable()) {
            getMethodNames(new Throwable().getStackTrace());
            Log.e(className, createLog(str), th);
        }
    }

    private static void getMethodNames(StackTraceElement[] stackTraceElementArr) {
        className = stackTraceElementArr[1].getFileName();
        methodName = stackTraceElementArr[1].getMethodName();
        lineNumber = stackTraceElementArr[1].getLineNumber();
    }

    public static void i(String str) {
        if (isDebuggable()) {
            getMethodNames(new Throwable().getStackTrace());
            Log.i(className, createLog(str));
        }
    }

    public static boolean isDebuggable() {
        return false;
    }

    public static void setmCallback(Callback callback) {
        mCallback = callback;
    }

    public static void v(String str) {
        if (isDebuggable()) {
            getMethodNames(new Throwable().getStackTrace());
            Log.v(className, createLog(str));
        }
    }

    public static void w(String str) {
        if (isDebuggable()) {
            getMethodNames(new Throwable().getStackTrace());
            Log.w(className, createLog(str));
        }
    }
}
