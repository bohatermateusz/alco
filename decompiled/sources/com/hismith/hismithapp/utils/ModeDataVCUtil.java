package com.hismith.hismithapp.utils;

import android.text.TextUtils;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class ModeDataVCUtil {
    public static final String BLOCK_FLAG = "MG";
    public static final String HEAD_PREFIX = "VC";
    public static final String HEAD_VERSION = "01";
    public static final int SERVER_INTERVAL_TIME = 100;
    public static final String TYPE_DOUBLE = "2";
    public static final String TYPE_LOOP = "0";
    public static final String TYPE_SERVO = "S";
    public static final String TYPE_SINGLE = "1";

    public static String buildHead(List<String> list) {
        StringBuilder sb = new StringBuilder("VC01T");
        sb.append(hex(100, 2)).append(Constans.DEVICE_TYPE_FUN);
        if (list != null && !list.isEmpty()) {
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) {
                    sb.append(",");
                }
                sb.append(list.get(i));
            }
        }
        return sb.toString();
    }

    public static String buildModeData(List<String> list, List<String> list2) {
        StringBuilder sb = new StringBuilder();
        if (list2 != null) {
            for (String str : list2) {
                if (!TextUtils.isEmpty(str)) {
                    sb.append(BLOCK_FLAG).append(normalizeBlock(str));
                }
            }
        }
        if (sb.length() > 0 && sb.charAt(sb.length() - 1) == 'G') {
            sb.deleteCharAt(sb.length() - 1);
        }
        return buildHead(list) + sb.toString();
    }

    public static String getContent(String str) {
        if (TextUtils.isEmpty(str)) {
            return "";
        }
        String upperCase = str.toUpperCase();
        int indexOf = upperCase.indexOf(BLOCK_FLAG);
        return (!upperCase.startsWith(HEAD_PREFIX) || indexOf <= 0) ? upperCase : upperCase.substring(indexOf);
    }

    public static String getHead(String str) {
        if (TextUtils.isEmpty(str)) {
            return "";
        }
        String upperCase = str.toUpperCase();
        int indexOf = upperCase.indexOf(BLOCK_FLAG);
        return (!upperCase.startsWith(HEAD_PREFIX) || indexOf <= 0) ? "" : upperCase.substring(0, indexOf);
    }

    public static int getVersion(String str) {
        if (TextUtils.isEmpty(str)) {
            return -1;
        }
        String upperCase = str.toUpperCase();
        if (upperCase.startsWith(HEAD_PREFIX) && upperCase.length() >= 4) {
            char charAt = upperCase.charAt(2);
            char charAt2 = upperCase.charAt(3);
            if (isHex(charAt) && isHex(charAt2)) {
                return Integer.parseInt(upperCase.substring(2, 4), 16);
            }
        }
        return -1;
    }

    public static String hex(int i, int i2) {
        String upperCase = Integer.toHexString(Math.max(0, i)).toUpperCase();
        while (upperCase.length() < i2) {
            upperCase = "0" + upperCase;
        }
        return upperCase.length() > i2 ? upperCase.substring(upperCase.length() - i2) : upperCase;
    }

    public static boolean isHex(char c) {
        if (c < '0' || c > '9') {
            return c >= 'A' && c <= 'F';
        }
        return true;
    }

    private static String normalizeBlock(String str) {
        String upperCase = str.toUpperCase();
        return upperCase.endsWith("G") ? upperCase : upperCase + "G";
    }

    public static List<String> parseFunCodes(String str) {
        ArrayList arrayList = new ArrayList();
        if (!TextUtils.isEmpty(str)) {
            String upperCase = str.toUpperCase();
            if (upperCase.startsWith(HEAD_PREFIX) && getVersion(upperCase) == 1) {
                return parseFunCodesV01(upperCase);
            }
        }
        return arrayList;
    }

    private static List<String> parseFunCodesV01(String str) {
        ArrayList arrayList = new ArrayList();
        if (str.length() >= 8 && str.charAt(7) == 'F') {
            String substring = str.substring(8);
            int indexOf = substring.indexOf(BLOCK_FLAG);
            if (indexOf >= 0) {
                substring = substring.substring(0, indexOf);
            }
            for (String str2 : substring.split(",")) {
                String trim = str2.trim();
                if (!TextUtils.isEmpty(trim)) {
                    arrayList.add(trim);
                }
            }
        }
        return arrayList;
    }

    public static List<String> splitBlocks(String str) {
        ArrayList arrayList = new ArrayList();
        String content = getContent(str);
        if (!TextUtils.isEmpty(content)) {
            if (!content.startsWith(BLOCK_FLAG)) {
                arrayList.add(content);
                return arrayList;
            }
            String[] split = content.split(BLOCK_FLAG);
            for (int i = 0; i < split.length; i++) {
                if (!TextUtils.isEmpty(split[i])) {
                    arrayList.add(split[i]);
                }
            }
        }
        return arrayList;
    }
}
