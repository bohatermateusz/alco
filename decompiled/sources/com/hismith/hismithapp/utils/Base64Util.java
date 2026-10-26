package com.hismith.hismithapp.utils;

import android.util.Base64;
import java.io.UnsupportedEncodingException;

/* loaded from: classes3.dex */
public class Base64Util {
    public static String getBase64(String str) {
        if (str == null) {
            return "";
        }
        try {
            return new String(Base64.encode(str.getBytes("utf-8"), 2), "utf-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return "";
        }
    }

    public static String getFromBase64(String str) {
        if (str == null) {
            return "";
        }
        try {
            return new String(Base64.decode(str, 2), "utf-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return "";
        }
    }
}
