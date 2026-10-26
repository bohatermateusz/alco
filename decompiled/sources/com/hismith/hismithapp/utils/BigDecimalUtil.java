package com.hismith.hismithapp.utils;

import java.math.BigDecimal;

/* loaded from: classes2.dex */
public class BigDecimalUtil {
    public static long intToLong(int i) {
        return new BigDecimal(i).longValue();
    }

    public static int longToInt(long j) {
        return new BigDecimal(j).intValue();
    }
}
