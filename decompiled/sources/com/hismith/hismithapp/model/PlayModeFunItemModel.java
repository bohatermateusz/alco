package com.hismith.hismithapp.model;

import java.util.Locale;

/* loaded from: classes3.dex */
public class PlayModeFunItemModel {
    public static final String FUN_PIU = "P";
    public static final String FUN_SPANK = "D";
    private String funCode;
    private int serialId;
    private int minute = 0;
    private int second = 0;
    private int hectoMs = 0;
    private int strength = 50;
    private int count = 1;
    private boolean isError = false;

    public PlayModeFunItemModel(String str) {
        this.funCode = str;
    }

    public int getCount() {
        return this.count;
    }

    public String getFunCode() {
        return this.funCode;
    }

    public int getHectoMs() {
        return this.hectoMs;
    }

    public int getMinute() {
        return this.minute;
    }

    public int getSecond() {
        return this.second;
    }

    public int getSerialId() {
        return this.serialId;
    }

    public int getStrength() {
        return this.strength;
    }

    public String getTimeText() {
        return String.format(Locale.US, "%02d:%02d.%03d", Integer.valueOf(this.minute), Integer.valueOf(this.second), Integer.valueOf(this.hectoMs));
    }

    public boolean isError() {
        return this.isError;
    }

    public boolean isSpank() {
        return FUN_SPANK.equalsIgnoreCase(this.funCode);
    }

    public void setCount(int i) {
        this.count = i;
    }

    public void setError(boolean z) {
        this.isError = z;
    }

    public void setFunCode(String str) {
        this.funCode = str;
    }

    public void setHectoMs(int i) {
        this.hectoMs = i;
    }

    public void setMinute(int i) {
        this.minute = i;
    }

    public void setSecond(int i) {
        this.second = i;
    }

    public void setSerialId(int i) {
        this.serialId = i;
    }

    public void setStrength(int i) {
        this.strength = i;
    }

    public String toString() {
        return "PlayModeFunItemModel{funCode='" + this.funCode + "', serialId=" + this.serialId + ", minute=" + this.minute + ", second=" + this.second + ", hectoMs=" + this.hectoMs + ", strength=" + this.strength + ", count=" + this.count + ", isError=" + this.isError + '}';
    }
}
