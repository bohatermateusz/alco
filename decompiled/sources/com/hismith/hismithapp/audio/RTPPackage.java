package com.hismith.hismithapp.audio;

import android.util.Base64;
import com.hismith.hismithapp.utils.ByteUtil;

/* loaded from: classes3.dex */
public class RTPPackage {
    private byte[] opusData;
    private String userId;
    private String voiceData;

    public RTPPackage(String str, String str2) {
        this.userId = str;
        this.voiceData = str2;
        this.opusData = Base64.decode(str2, 2);
    }

    public byte[] getOpusData() {
        return this.opusData;
    }

    public short[] getOpusDataByShort() {
        byte[] bArr = this.opusData;
        if (bArr == null) {
            return null;
        }
        return ByteUtil.toShortArray(bArr);
    }

    public String getUserId() {
        return this.userId;
    }

    public String getVoiceData() {
        return this.voiceData;
    }

    public void setOpusData(byte[] bArr) {
        this.opusData = bArr;
    }

    public void setUserId(String str) {
        this.userId = str;
    }

    public void setVoiceData(String str) {
        this.voiceData = str;
    }
}
