package com.hismith.hismithapp.utils;

import java.nio.charset.StandardCharsets;

/* loaded from: classes3.dex */
public class OrdUtil {
    public static int ord(char c) {
        return c < 128 ? c : ord(Character.toString(c));
    }

    public static int ord(String str) {
        if (str.length() > 0) {
            return str.getBytes(StandardCharsets.UTF_8)[0] & 255;
        }
        return 0;
    }
}
