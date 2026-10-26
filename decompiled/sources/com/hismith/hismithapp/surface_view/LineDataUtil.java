package com.hismith.hismithapp.surface_view;

import android.text.TextUtils;
import androidx.exifinterface.media.ExifInterface;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class LineDataUtil {
    private static int serverIntervalTime = 100;
    private static long startTime;

    private static String buildBodyData(String str, String str2) throws Exception {
        StringBuilder sb = new StringBuilder();
        String str3 = "";
        try {
            String[] split = str.split("\\$");
            String[] split2 = str2.split("G");
            for (int i = 0; i < split2.length; i++) {
                if (split2[i].indexOf("H") == 0) {
                    String str4 = split2[i];
                    int i2 = 1;
                    int intValue = Integer.valueOf(str4.substring(1, str4.indexOf(",")), 16).intValue();
                    try {
                        String str5 = split2[i];
                        i2 = Integer.valueOf(str5.substring(str5.indexOf(",") + 1), 16).intValue();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    for (int i3 = 0; i3 < i2; i3++) {
                        str3 = "head:" + split[intValue];
                        sb.append(buildToSpeed(split, intValue));
                    }
                } else {
                    str3 = "body:" + split2[i];
                    sb.append(buildToSpeed(split2, i));
                }
            }
            return sb.toString();
        } catch (Exception e2) {
            throw new Exception("buildToSpeed 数据有错:" + str3 + ", /" + e2.getMessage());
        }
    }

    public static String[] buildServerToClientData(String str) {
        return buildServerToClientData(str, -1);
    }

    public static String[] buildServerToClientData(String str, int i) {
        char c;
        String[] strArr = new String[1];
        if (!TextUtils.isEmpty(str)) {
            startTime = System.currentTimeMillis();
            try {
                int intValue = Integer.valueOf(str.substring(str.indexOf(ExifInterface.GPS_MEASUREMENT_INTERRUPTED) + 1, str.indexOf(ExifInterface.GPS_DIRECTION_TRUE)), 16).intValue();
                if (intValue == 1) {
                    String[] split = str.split("\\*");
                    if (split.length == 3) {
                        String str2 = split[0];
                        return new String[]{Integer.valueOf(str2.substring(str2.indexOf(ExifInterface.GPS_DIRECTION_TRUE) + 1), 16).intValue() + "", buildBodyData(split[1], split[2]), ""};
                    }
                    if (split.length == 5) {
                        String str3 = split[0];
                        return new String[]{Integer.valueOf(str3.substring(str3.indexOf(ExifInterface.GPS_DIRECTION_TRUE) + 1), 16).intValue() + "", buildBodyData(split[1], split[2]), buildBodyData(split[3], split[4])};
                    }
                } else if (intValue == 2) {
                    strArr = new String[5];
                    String[] split2 = str.split("\\*");
                    LogUtil.e("dataArr.length:" + split2.length);
                    if (split2.length == 9) {
                        String str4 = split2[0];
                        String str5 = split2[1];
                        String str6 = split2[2];
                        String str7 = split2[3];
                        String str8 = split2[4];
                        String str9 = split2[5];
                        c = 0;
                        try {
                            String str10 = split2[6];
                            String str11 = split2[7];
                            String str12 = split2[8];
                            strArr[0] = Integer.valueOf(str4.substring(str4.indexOf(ExifInterface.GPS_DIRECTION_TRUE) + 1), 16).intValue() + "";
                            strArr[1] = buildBodyData(str5, str6);
                            strArr[2] = buildBodyData(str7, str8);
                            strArr[3] = buildBodyData(str9, str10);
                            strArr[4] = buildBodyData(str11, str12);
                            return strArr;
                        } catch (Exception e) {
                            e = e;
                            e.printStackTrace();
                            LogUtil.e("error错误：" + i);
                            LogUtil.e("error错误数据：" + str);
                            LogUtil.e("error错误地方：" + e.getMessage());
                            strArr[c] = "";
                            return strArr;
                        }
                    }
                }
            } catch (Exception e2) {
                e = e2;
                c = 0;
            }
        }
        return strArr;
    }

    public static String buildToServerData(String str, int i) {
        if (TextUtils.isEmpty(str)) {
            return "";
        }
        String[] split = str.split("G");
        StringBuilder sb = new StringBuilder();
        int i2 = serverIntervalTime / i;
        String str2 = "";
        int i3 = i2;
        int i4 = 1;
        for (int i5 = 0; i5 < split.length; i5++) {
            String substring = split[i5].substring(0, 2);
            if (Integer.toHexString(255).equalsIgnoreCase(substring) || Integer.toHexString(Constans.START_TYPE_PIU).equalsIgnoreCase(substring)) {
                if (!TextUtils.isEmpty(str2)) {
                    sb.append(str2);
                    sb.append(Integer.toHexString(i4));
                    sb.append("G");
                    str2 = "";
                    i4 = 1;
                }
                sb.append(split[i5]);
                sb.append("G");
            } else {
                i3--;
                if (i3 == 0) {
                    if (!TextUtils.isEmpty(str2)) {
                        if (str2.equals(split[i5])) {
                            i4++;
                        } else {
                            sb.append(str2);
                            sb.append(Integer.toHexString(i4));
                            sb.append("G");
                            i4 = 1;
                        }
                    }
                    str2 = split[i5];
                    i3 = i2;
                }
            }
        }
        if (!TextUtils.isEmpty(str2)) {
            sb.append(str2);
            sb.append(Integer.toHexString(i4));
            sb.append("G");
        }
        LogUtil.e("serverBuild:" + ((Object) sb));
        return sb.toString();
    }

    public static String buildToServerData2(String str, int i) {
        String str2 = "";
        if (TextUtils.isEmpty(str)) {
            return "";
        }
        StringBuilder sb = new StringBuilder(str);
        StringBuilder sb2 = new StringBuilder();
        int i2 = serverIntervalTime / i;
        int i3 = i2 - 1;
        int i4 = 1;
        while (i3 < sb.length() / 3) {
            int i5 = i3 * 3;
            String substring = sb.substring(i5, i5 + 2);
            if (!TextUtils.isEmpty(str2)) {
                if (str2.equals(substring)) {
                    i4++;
                } else {
                    sb2.append(str2);
                    sb2.append(Integer.toHexString(i4));
                    sb2.append("G");
                    i4 = 1;
                }
            }
            i3 += i2;
            str2 = substring;
        }
        sb2.append(str2);
        sb2.append(Integer.toHexString(i4));
        sb2.append("G");
        LogUtil.e("serverBuild:" + ((Object) sb2));
        return sb2.toString();
    }

    public static String buildToServerStructure(int i, String... strArr) {
        if (strArr == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder(ExifInterface.GPS_MEASUREMENT_INTERRUPTED);
        String hexString = Integer.toHexString(i);
        if (hexString.length() < 2) {
            hexString = "0" + hexString;
        }
        sb.append(hexString);
        sb.append(ExifInterface.GPS_DIRECTION_TRUE);
        sb.append(Integer.toHexString(serverIntervalTime));
        for (String str : strArr) {
            if (!TextUtils.isEmpty(str)) {
                sb.append("**");
                sb.append(str);
            }
        }
        return sb.toString();
    }

    private static String buildToSpeed(String[] strArr, int i) throws Exception {
        String str = "";
        if (strArr == null || strArr.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        try {
            String[] split = strArr[i].split("G");
            for (int i2 = 0; i2 < split.length; i2++) {
                if (split[i2].indexOf("H") == 0) {
                    StringBuilder sb2 = new StringBuilder();
                    String str2 = split[i2];
                    sb2.append(str2.substring(1, str2.indexOf(","))).append(", ").append(i2).toString();
                    String str3 = split[i2];
                    int intValue = Integer.valueOf(str3.substring(1, str3.indexOf(",")), 16).intValue();
                    StringBuilder sb3 = new StringBuilder();
                    String str4 = split[i2];
                    str = sb3.append(str4.substring(str4.indexOf(",") + 1)).append(", ").append(i2).toString();
                    try {
                        String str5 = split[i2];
                        r10 = Integer.valueOf(str5.substring(str5.indexOf(",") + 1), 16).intValue();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    for (int i3 = 0; i3 < r10; i3++) {
                        sb.append(buildToSpeed(strArr, intValue));
                    }
                } else {
                    String str6 = split[i2].substring(0, 2) + "," + i2;
                    String substring = split[i2].substring(0, 2);
                    Integer.valueOf(substring, 16).intValue();
                    str = split[i2].substring(2) + "," + i2;
                    r10 = TextUtils.isEmpty(split[i2].substring(2)) ? 1 : Integer.valueOf(split[i2].substring(2), 16).intValue();
                    if (!Integer.toHexString(255).equalsIgnoreCase(substring) && !Integer.toHexString(Constans.START_TYPE_PIU).equalsIgnoreCase(substring)) {
                        for (int i4 = 0; i4 < r10; i4++) {
                            sb.append(substring);
                            sb.append("G");
                        }
                    }
                    sb.append(split[i2].toUpperCase());
                    sb.append("G");
                }
            }
            return sb.toString();
        } catch (Exception unused) {
            throw new Exception("buildToSpeed 数据有错:" + str + "。headArr:" + strArr[i]);
        }
    }

    public static String buildToUIData(String str, int i, int i2) {
        int intValue;
        if (TextUtils.isEmpty(str)) {
            return "";
        }
        String[] split = str.split("G");
        StringBuilder sb = new StringBuilder();
        int i3 = i2 / i;
        int i4 = 0;
        for (int i5 = 0; i5 < split.length; i5++) {
            int intValue2 = Integer.valueOf(split[i5].substring(0, 2), 16).intValue();
            if (intValue2 == 255 || intValue2 == 254) {
                sb.append(split[i5]);
                sb.append("G");
                i4++;
            } else {
                if (i5 == 0) {
                    intValue = intValue2;
                } else {
                    intValue = Integer.valueOf(split[(i5 - 1) - i4].substring(0, 2), 16).intValue();
                    i4 = 0;
                }
                float f = 1.0f;
                while (true) {
                    float f2 = i3;
                    if (f <= f2) {
                        try {
                            String str2 = Integer.toHexString((int) (intValue - ((intValue - intValue2) * (f / f2)))) + "G";
                            if (str2.length() < 3) {
                                str2 = "0" + str2;
                            }
                            sb.append(str2);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        f += 1.0f;
                    }
                }
            }
        }
        return sb.toString();
    }
}
