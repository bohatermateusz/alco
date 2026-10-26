package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class MessageVoiceModel {
    private String dat;
    private int rate;

    public String getDat() {
        return this.dat;
    }

    public int getRate() {
        return this.rate;
    }

    public void setDat(String str) {
        this.dat = str;
    }

    public void setRate(int i) {
        this.rate = i;
    }

    public String toString() {
        return "MessageVoiceModel{rate=" + this.rate + ", dat= " + this.dat.length() + '}';
    }
}
