package com.hismith.hismithapp.utils;

import java.util.ArrayList;

/* loaded from: classes3.dex */
public class ModeTimeUtil {
    public static String formatMilliTime(int i) {
        int i2 = i / 1000;
        StringBuilder sb = new StringBuilder();
        int i3 = i2 % 60;
        int i4 = i2 / 60;
        int i5 = i4 % 60;
        int i6 = i4 / 60;
        if (i6 > 0) {
            if (i6 < 10) {
                sb.append("0");
            }
            sb.append(i6 + ":");
        }
        if (i5 < 10) {
            sb.append("0");
        }
        sb.append(i5 + ":");
        if (i3 < 10) {
            sb.append("0");
        }
        sb.append(i3);
        return sb.toString();
    }

    public static String formatMilliTime(long j) {
        long j2 = j / 1000;
        StringBuilder sb = new StringBuilder();
        long j3 = j2 % 60;
        long j4 = j2 / 60;
        long j5 = j4 % 60;
        long j6 = j4 / 60;
        if (j6 > 0) {
            if (j6 < 10) {
                sb.append("0");
            }
            sb.append(j6 + ":");
        }
        if (j5 < 10) {
            sb.append("0");
        }
        sb.append(j5 + ":");
        if (j3 < 10) {
            sb.append("0");
        }
        sb.append(j3);
        return sb.toString();
    }

    public static String formatSecondTime(int i) {
        StringBuilder sb = new StringBuilder();
        int i2 = i % 60;
        int i3 = i / 60;
        int i4 = i3 % 60;
        int i5 = i3 / 60;
        if (i5 > 0) {
            if (i5 < 10) {
                sb.append("0");
            }
            sb.append(i5 + ":");
        }
        if (i4 < 10) {
            sb.append("0");
        }
        sb.append(i4 + ":");
        if (i2 < 10) {
            sb.append("0");
        }
        sb.append(i2);
        return sb.toString();
    }

    public static String formatTime(int i) {
        int i2 = i / 100;
        StringBuilder sb = new StringBuilder();
        int i3 = i2 % 60;
        int i4 = i2 / 60;
        int i5 = i4 % 60;
        int i6 = i4 / 60;
        if (i6 > 0) {
            if (i6 < 10) {
                sb.append("0");
            }
            sb.append(i6 + ":");
        }
        if (i5 < 10) {
            sb.append("0");
        }
        sb.append(i5 + ":");
        if (i3 < 10) {
            sb.append("0");
        }
        sb.append(i3);
        return sb.toString();
    }

    public static int getTotalTime(String str) {
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        int i6 = 0;
        if (str == null) {
            return 0;
        }
        String[] split = str.split("G");
        ArrayList arrayList = new ArrayList();
        ArrayList arrayList2 = new ArrayList();
        int i7 = 0;
        while (true) {
            i = 3;
            i2 = 1;
            i3 = 16;
            if (i7 >= split.length) {
                break;
            }
            String substring = split[i7].substring(0, 1);
            Integer.valueOf(split[i7].substring(1, 3), 16).intValue();
            if (!"0".equals(substring)) {
                arrayList2.add(split[i7]);
            }
            i7++;
        }
        int i8 = 0;
        while (i8 < split.length) {
            String str2 = split[i8];
            String substring2 = str2.substring(i6, i2);
            if ("S".equals(substring2)) {
                arrayList.add(Integer.valueOf((Integer.valueOf(str2.substring(10, 12), i3).intValue() * 100) + (Integer.valueOf(str2.substring(12, 13), i3).intValue() * 10)));
            } else if ("2".equals(substring2)) {
                arrayList.add(Integer.valueOf((Integer.valueOf(str2.substring(9, 11), i3).intValue() * 100) + (Integer.valueOf(str2.substring(11, 12), i3).intValue() * 10)));
            } else {
                int i9 = 5;
                if ("1".equals(substring2)) {
                    arrayList.add(Integer.valueOf((Integer.valueOf(str2.substring(5, 7), i3).intValue() * 100) + (Integer.valueOf(str2.substring(7, 8), i3).intValue() * 10)));
                } else if ("0".equals(substring2)) {
                    int intValue = Integer.valueOf(str2.substring(5, 7), i3).intValue();
                    int i10 = 0;
                    while (i10 < intValue) {
                        int intValue2 = Integer.valueOf(str2.substring(i2, i), i3).intValue();
                        int intValue3 = Integer.valueOf(str2.substring(i, i9), i3).intValue();
                        while (intValue2 <= intValue3) {
                            String str3 = (String) arrayList2.get(intValue2 - 1);
                            String[] strArr = split;
                            String substring3 = str3.substring(0, 1);
                            if ("1".equals(substring3)) {
                                i3 = 16;
                                i4 = intValue2;
                                i5 = (Integer.valueOf(str3.substring(5, 7), 16).intValue() * 100) + (Integer.valueOf(str3.substring(7, 8), 16).intValue() * 10);
                            } else {
                                i4 = intValue2;
                                i3 = 16;
                                if ("2".equals(substring3)) {
                                    i5 = (Integer.valueOf(str3.substring(9, 11), 16).intValue() * 100) + (Integer.valueOf(str3.substring(11, 12), 16).intValue() * 10);
                                } else if ("S".equals(substring3)) {
                                    i5 = (Integer.valueOf(str3.substring(12, 13), 16).intValue() * 10) + (Integer.valueOf(str3.substring(10, 12), 16).intValue() * 100);
                                } else {
                                    i5 = 0;
                                }
                            }
                            arrayList.add(Integer.valueOf(i5));
                            intValue2 = i4 + 1;
                            split = strArr;
                        }
                        i10++;
                        split = split;
                        i = 3;
                        i2 = 1;
                        i9 = 5;
                    }
                }
            }
            i8++;
            split = split;
            i6 = 0;
            i = 3;
            i2 = 1;
        }
        int i11 = 0;
        for (int i12 = 0; i12 < arrayList.size(); i12++) {
            i11 += ((Integer) arrayList.get(i12)).intValue();
        }
        return i11;
    }
}
