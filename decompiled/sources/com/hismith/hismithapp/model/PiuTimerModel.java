package com.hismith.hismithapp.model;

import java.io.Serializable;

/* loaded from: classes3.dex */
public class PiuTimerModel implements Serializable {
    private int second;
    private long updateTimeMillis;

    public PiuTimerModel(int i) {
        this.updateTimeMillis = 0L;
        this.second = i;
        this.updateTimeMillis = System.currentTimeMillis();
    }

    public int getSecond() {
        return this.second;
    }

    public long getUpdateTimeMillis() {
        return this.updateTimeMillis;
    }

    public void setSecond(int i) {
        this.second = i;
        this.updateTimeMillis = System.currentTimeMillis();
    }

    public void setUpdateTimeMillis(long j) {
        this.updateTimeMillis = j;
    }

    public String toString() {
        return "PiuTimerModel{second=" + this.second + '}';
    }
}
