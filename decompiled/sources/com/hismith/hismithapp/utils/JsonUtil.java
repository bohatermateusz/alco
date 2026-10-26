package com.hismith.hismithapp.utils;

import android.content.Context;
import com.google.gson.Gson;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Type;

/* loaded from: classes3.dex */
public class JsonUtil {
    private JsonUtil() {
    }

    public static <T> T fromJson(String str, Class<T> cls) {
        return (T) new Gson().fromJson(str, (Class) cls);
    }

    public static <T> T fromJson(String str, Type type) {
        return (T) new Gson().fromJson(str, type);
    }

    public static String getAsstesJson(Context context, String str) {
        StringBuilder sb = new StringBuilder();
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(context.getAssets().open(str), "ISO-8859-1"));
            while (true) {
                String readLine = bufferedReader.readLine();
                if (readLine == null) {
                    break;
                }
                sb.append(readLine.trim());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sb.toString();
    }

    public static String toJson(Object obj) {
        return new Gson().toJson(obj);
    }
}
