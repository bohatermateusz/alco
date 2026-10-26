package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class RemoteChatModel {
    private String message;
    private int send;

    public RemoteChatModel(String str, int i) {
        this.message = str;
        this.send = i;
    }

    public String getMessage() {
        return this.message;
    }

    public int getSend() {
        return this.send;
    }

    public void setMessage(String str) {
        this.message = str;
    }

    public void setSend(int i) {
        this.send = i;
    }
}
