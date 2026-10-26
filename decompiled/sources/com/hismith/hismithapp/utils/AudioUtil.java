package com.hismith.hismithapp.utils;

import android.os.Environment;
import androidx.core.view.MotionEventCompat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class AudioUtil {
    public static int averValue(byte[] bArr, int i) {
        int i2 = i / 8;
        int i3 = 0;
        for (int i4 = 0; i4 < bArr.length / i2; i4++) {
            int i5 = i4 * 2;
            int i6 = (bArr[i5] & 255) + (bArr[i5 + 1] & MotionEventCompat.ACTION_POINTER_INDEX_MASK);
            if (i6 >= 32767) {
                i6 = 65535 - i6;
            }
            i3 += i6;
        }
        return i3 / (bArr.length / i2);
    }

    public static void getAver() {
        List<byte[]> readFile = MuteUtil.readFile(Environment.getExternalStorageDirectory().getAbsolutePath() + "/AudiioRecordFile", "aaa.wav");
        for (int i = 0; i < readFile.size(); i++) {
            getAvg(readFile.get(i), 16);
        }
    }

    public static int getAvg(byte[] bArr, int i) {
        double d = 2.0d;
        double pow = Math.pow(2.0d, i);
        int i2 = i / 8;
        ArrayList arrayList = new ArrayList();
        double[] dArr = new double[i2];
        for (int i3 = 0; i3 < i2; i3++) {
            dArr[i3] = Math.pow(256.0d, i3);
        }
        int length = bArr.length / i2;
        int i4 = 0;
        while (i4 < length) {
            int i5 = i4 * i2;
            double d2 = 0.0d;
            int i6 = 0;
            while (i6 < i2) {
                double d3 = d;
                d2 += (bArr[i5 + i6] & 255) * (i6 > 0 ? dArr[i6] : 1.0d);
                if (i6 == i2 - 1 && d2 >= pow / d3) {
                    d2 = pow - d2;
                }
                i6++;
                d = d3;
            }
            arrayList.add(Double.valueOf(d2));
            i4++;
            d = d;
        }
        float f = 0.0f;
        for (int i7 = 0; i7 < arrayList.size(); i7++) {
            f = (float) (f + ((Double) arrayList.get(i7)).doubleValue());
        }
        return Math.round(f / arrayList.size());
    }

    public static double getMax(byte[] bArr, int i) {
        double d;
        double d2 = 2.0d;
        double pow = Math.pow(2.0d, i);
        int i2 = i / 8;
        ArrayList arrayList = new ArrayList();
        double[] dArr = new double[i2];
        for (int i3 = 0; i3 < i2; i3++) {
            dArr[i3] = Math.pow(256.0d, i3);
        }
        int length = bArr.length / i2;
        int i4 = 0;
        while (true) {
            d = 0.0d;
            if (i4 >= length) {
                break;
            }
            int i5 = i4 * i2;
            int i6 = 0;
            while (i6 < i2) {
                double d3 = d2;
                d += (bArr[i5 + i6] & 255) * (i6 > 0 ? dArr[i6] : 1.0d);
                if (i6 == i2 - 1 && d >= pow / d3) {
                    d = pow - d;
                }
                i6++;
                d2 = d3;
            }
            arrayList.add(Double.valueOf(d));
            i4++;
            d2 = d2;
        }
        for (int i7 = 0; i7 < arrayList.size(); i7++) {
            if (d < ((Double) arrayList.get(i7)).doubleValue()) {
                d = ((Double) arrayList.get(i7)).doubleValue();
            }
        }
        return d;
    }

    public static boolean isMute(byte[] bArr, int i) {
        double d;
        LogUtil.i("data:" + StringUtils.bytesToHex1(bArr));
        double d2 = 2.0d;
        double pow = Math.pow(2.0d, i);
        int i2 = i / 8;
        double d3 = i2;
        double pow2 = Math.pow(8.0d, d3) * d3;
        ArrayList arrayList = new ArrayList();
        double[] dArr = new double[i2];
        for (int i3 = 0; i3 < i2; i3++) {
            dArr[i3] = Math.pow(256.0d, i3);
        }
        int length = bArr.length / i2;
        int i4 = 0;
        while (true) {
            d = 0.0d;
            if (i4 >= length) {
                break;
            }
            int i5 = i4 * i2;
            double d4 = d2;
            int i6 = 0;
            while (i6 < i2) {
                int i7 = length;
                d += (bArr[i5 + i6] & 255) * (i6 > 0 ? dArr[i6] : 1.0d);
                if (i6 == i2 - 1 && d >= pow / d4) {
                    d = pow - d;
                }
                i6++;
                length = i7;
            }
            arrayList.add(Double.valueOf(d));
            i4++;
            d2 = d4;
        }
        LogUtil.i("meanArr:" + arrayList.toString());
        for (int i8 = 0; i8 < arrayList.size(); i8++) {
            d += ((Double) arrayList.get(i8)).doubleValue();
        }
        double size = d / arrayList.size();
        LogUtil.i("静音值比较：mean(" + size + ") <= mute(" + pow2 + ")");
        return size <= pow2;
    }

    public static int searchThreshold(byte[] bArr) {
        int length = bArr.length / 2;
        short[] sArr = new short[length];
        ByteBuffer.wrap(bArr).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(sArr);
        for (int i = 0; i < length; i++) {
            short s = sArr[i];
            if (s >= 15000 || s <= -15000) {
                return i;
            }
        }
        return -1;
    }
}
