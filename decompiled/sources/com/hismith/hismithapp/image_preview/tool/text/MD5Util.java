package com.hismith.hismithapp.image_preview.tool.text;

import androidx.exifinterface.media.ExifInterface;
import androidx.media3.exoplayer.upstream.CmcdData;
import com.twitter.sdk.android.core.BuildConfig;
import java.security.MessageDigest;

/* loaded from: classes3.dex */
public class MD5Util {
    private static final String[] hexDigIts = {"0", "1", "2", ExifInterface.GPS_MEASUREMENT_3D, "4", "5", "6", "7", "8", BuildConfig.BUILD_NUMBER, CmcdData.OBJECT_TYPE_AUDIO_ONLY, "b", "c", "d", "e", "f"};

    public static String byteArrayToHexString(byte[] bArr) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bArr) {
            sb.append(byteToHexString(b));
        }
        return sb.toString();
    }

    public static String byteToHexString(byte b) {
        int i = b;
        if (b < 0) {
            i = b + 256;
        }
        int i2 = i / 16;
        StringBuilder sb = new StringBuilder();
        String[] strArr = hexDigIts;
        return sb.append(strArr[i2]).append(strArr[i % 16]).toString();
    }

    public static String md5Encode(String str) {
        return md5Encode(str, "utf-8");
    }

    public static String md5Encode(String str, String str2) {
        try {
            MessageDigest messageDigest = MessageDigest.getInstance("MD5");
            if (str2 != null && !"".equals(str2)) {
                return byteArrayToHexString(messageDigest.digest(str.getBytes(str2)));
            }
            return byteArrayToHexString(messageDigest.digest(str.getBytes()));
        } catch (Exception e) {
            e.printStackTrace();
            return str;
        }
    }
}
