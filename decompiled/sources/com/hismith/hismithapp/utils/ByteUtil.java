package com.hismith.hismithapp.utils;

/* loaded from: classes3.dex */
public class ByteUtil {
    public static byte[] mergeArr(byte[] bArr, byte[] bArr2) {
        byte[] bArr3 = new byte[bArr.length + bArr2.length];
        System.arraycopy(bArr, 0, bArr3, 0, bArr.length);
        System.arraycopy(bArr2, 0, bArr3, bArr.length, bArr2.length);
        return bArr3;
    }

    public static byte[] remoteArr(byte[] bArr, int i) {
        if (bArr.length < i) {
            return bArr;
        }
        int length = bArr.length - i;
        byte[] bArr2 = new byte[length];
        System.arraycopy(bArr, i, bArr2, 0, length);
        return bArr2;
    }

    public static short[] toShortArray(byte[] bArr) {
        int length = bArr.length >> 1;
        short[] sArr = new short[length];
        for (int i = 0; i < length; i++) {
            int i2 = i * 2;
            sArr[i] = (short) ((bArr[i2 + 1] & 255) | (bArr[i2] << 8));
        }
        return sArr;
    }
}
