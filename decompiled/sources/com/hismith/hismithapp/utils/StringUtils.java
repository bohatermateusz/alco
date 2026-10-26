package com.hismith.hismithapp.utils;

import android.text.TextUtils;
import com.google.common.base.Ascii;
import java.util.List;

/* loaded from: classes3.dex */
public class StringUtils {
    private static final char[] HEX_CHAR = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

    public static String bytesToHex1(byte[] bArr) {
        char[] cArr = new char[bArr.length * 2];
        int i = 0;
        for (int i2 : bArr) {
            if (i2 < 0) {
                i2 += 256;
            }
            int i3 = i + 1;
            char[] cArr2 = HEX_CHAR;
            cArr[i] = cArr2[i2 / 16];
            i += 2;
            cArr[i3] = cArr2[i2 % 16];
        }
        return new String(cArr);
    }

    public static String bytesToHex2(byte[] bArr) {
        char[] cArr = new char[bArr.length * 2];
        int i = 0;
        for (byte b : bArr) {
            int i2 = i + 1;
            char[] cArr2 = HEX_CHAR;
            cArr[i] = cArr2[(b >>> 4) & 15];
            i += 2;
            cArr[i2] = cArr2[b & Ascii.SI];
        }
        return new String(cArr);
    }

    public static String bytesToHex3(byte[] bArr) {
        StringBuilder sb = new StringBuilder(bArr.length * 2);
        for (byte b : bArr) {
            sb.append(String.format("%02x", new Integer(b & 255)));
        }
        return sb.toString();
    }

    public static boolean contains(String[] strArr, String str) {
        if (strArr != null && strArr.length > 0 && !TextUtils.isEmpty(str)) {
            for (String str2 : strArr) {
                if (str2.equals(str)) {
                    return true;
                }
            }
        }
        return false;
    }

    public static int countString(String str, String str2) {
        String replaceAll = str.replaceAll(str2, "");
        return (str.length() - replaceAll.length()) / str2.length();
    }

    public static String[] cut(String str, int i) {
        if (TextUtils.isEmpty(str)) {
            return new String[0];
        }
        int length = (str.length() / i) + (str.length() % i > 0 ? 1 : 0);
        String[] strArr = new String[length];
        for (int i2 = 0; i2 < length; i2++) {
            int i3 = i2 * i;
            int i4 = i3 + i;
            if (i4 > str.length()) {
                i4 = str.length();
            }
            strArr[i2] = str.substring(i3, i4);
        }
        return strArr;
    }

    public static String getFormatNum(String str) {
        return str + "";
    }

    public static String join(List<String> list, String str) {
        if (list == null || list.size() == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            sb.append(list.get(i));
            if (i < list.size() - 1) {
                sb.append(str);
            }
        }
        return sb.toString();
    }

    public static String join(String[] strArr, String str) {
        if (strArr == null || strArr.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < strArr.length; i++) {
            sb.append(strArr[i]);
            if (i < strArr.length - 1) {
                sb.append(str);
            }
        }
        return sb.toString();
    }

    public static byte[] toBytes(String str) {
        if (str == null || str.trim().equals("")) {
            return new byte[0];
        }
        byte[] bArr = new byte[str.length() / 2];
        for (int i = 0; i < str.length() / 2; i++) {
            int i2 = i * 2;
            bArr[i] = (byte) Integer.parseInt(str.substring(i2, i2 + 2), 16);
        }
        return bArr;
    }
}
