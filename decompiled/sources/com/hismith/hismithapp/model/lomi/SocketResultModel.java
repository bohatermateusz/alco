package com.hismith.hismithapp.model.lomi;

/* loaded from: classes3.dex */
public class SocketResultModel {
    private String code;
    private LomiMessage[] message;
    private String status;

    public String getCode() {
        return this.code;
    }

    public LomiMessage[] getMessage() {
        return this.message;
    }

    public String getStatus() {
        return this.status;
    }

    public void setCode(String str) {
        this.code = str;
    }

    public void setMessage(LomiMessage[] lomiMessageArr) {
        this.message = lomiMessageArr;
    }

    public void setStatus(String str) {
        this.status = str;
    }
}
