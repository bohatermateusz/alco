package com.hismith.hismithapp.utils;

import android.util.Log;

/* loaded from: classes3.dex */
public class LogUtil {
    private static final int MAX_LOG_LENGTH = 4000;
    static String className;
    static int lineNumber;
    static String methodName;

    private static String createLog(String str) {
        StringBuffer stringBuffer = new StringBuffer("================");
        stringBuffer.append(methodName);
        stringBuffer.append("(").append(className).append(":").append(lineNumber).append(")================:");
        if (str != null && str.length() > 0) {
            if (str.length() > 1024) {
                int length = str.length();
                int i = 0;
                while (i < length) {
                    int min = Math.min(i + 1024, length);
                    stringBuffer.append(str.substring(i, min));
                    if (min < length) {
                        stringBuffer.append("\n");
                    }
                    i = min;
                }
            } else {
                stringBuffer.append(str);
            }
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

    public static void printLongLog(String str, String str2) {
        if (str2.length() <= MAX_LOG_LENGTH) {
            Log.d(str, str2);
        } else {
            Log.d(str, str2.substring(0, MAX_LOG_LENGTH));
            printLongLog(str, str2.substring(MAX_LOG_LENGTH));
        }
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
