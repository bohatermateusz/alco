package com.hismith.hismithapp.utils;

import android.text.TextUtils;
import android.text.format.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;

/* loaded from: classes3.dex */
public class TimeUtil {
    private static final String TIME_MAX = "23:59:59";
    private static final String TIME_ZERO = "00:00";
    private static final String YMDHMS_FORMAT = "yyyy-MM-dd HH:mm:ss";
    private static final String search_DateFormat = "MM/dd/yyyy HH:mm:ss";
    private static final int seconds_of_1hour = 3600;
    private static final int seconds_of_1minute = 60;
    private static final int seconds_of_2hour = 7200;
    private static final int seconds_of_3hour = 10800;

    public static String Local2UTC() {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(YMDHMS_FORMAT);
        simpleDateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));
        return simpleDateFormat.format(new Date());
    }

    public static String getCurrentTimeStamp() {
        return String.valueOf(System.currentTimeMillis() / 1000);
    }

    public static String getUSDateTimeFormat(long j) {
        return new SimpleDateFormat("HH:mm, MMMM dd, yyyy", Locale.US).format(new Date(j));
    }

    public static String longToString(long j, String str) {
        if (TextUtils.isEmpty(str)) {
            str = YMDHMS_FORMAT;
        }
        return new SimpleDateFormat(str).format(new Date(j));
    }

    public static String searchTimeFormat(String str) {
        if (TextUtils.isEmpty(str)) {
            return "";
        }
        try {
            return (String) DateFormat.format(YMDHMS_FORMAT, searchTimeToLong(str));
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    public static long searchTimeToLong(String str) {
        if (TextUtils.isEmpty(str)) {
            return 0L;
        }
        try {
            String[] split = str.split(" ");
            String str2 = split[0] + " " + split[1];
            int i = "pm".equals(split[2]) ? 43200000 : 0;
            SimpleDateFormat simpleDateFormat = new SimpleDateFormat(search_DateFormat);
            simpleDateFormat.setTimeZone(TimeZone.getDefault());
            return simpleDateFormat.parse(str2).getTime() + i;
        } catch (ParseException e) {
            e.printStackTrace();
            return 0L;
        }
    }

    public static String secondsToTime(int i) {
        if (i <= 0) {
            return TIME_ZERO;
        }
        int i2 = i / 60;
        if (i2 < 60) {
            return unitFormat(i2) + ":" + unitFormat(i % 60);
        }
        int i3 = i2 / 60;
        if (i3 > 23) {
            return TIME_MAX;
        }
        int i4 = i2 % 60;
        return unitFormat(i3) + ":" + unitFormat(i4) + ":" + unitFormat((i - (i3 * seconds_of_1hour)) - (i4 * 60));
    }

    public static Date stringConvertDate(String str) {
        try {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.CHINA).parse(str);
        } catch (ParseException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static String timeStamp2Date(long j, String str) {
        if (str == null || str.isEmpty()) {
            str = YMDHMS_FORMAT;
        }
        return new SimpleDateFormat(str).format(new Date(j));
    }

    public static String unitFormat(int i) {
        try {
            return (i < 0 || i >= 10) ? "" + i : "0" + Integer.toString(i);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static String utc2Local(String str) {
        Date date;
        try {
            if (TextUtils.isEmpty(str)) {
                return "";
            }
            SimpleDateFormat simpleDateFormat = new SimpleDateFormat(YMDHMS_FORMAT);
            simpleDateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));
            try {
                date = simpleDateFormat.parse(str);
            } catch (ParseException e) {
                e.printStackTrace();
                date = null;
            }
            SimpleDateFormat simpleDateFormat2 = new SimpleDateFormat(YMDHMS_FORMAT);
            simpleDateFormat2.setTimeZone(TimeZone.getDefault());
            return simpleDateFormat2.format(Long.valueOf(date.getTime()));
        } catch (Exception e2) {
            e2.printStackTrace();
            return "";
        }
    }
}
