package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class PlayModeItemModel {
    private int serialId;
    private String single;
    private int start = 0;
    private int end = 0;
    private int start2 = 0;
    private int end2 = 0;
    private int seconds = 0;
    private int milli = 0;
    private int count = 1;
    private int smooth = 10;
    private int funData0 = 0;
    private int funData1 = 0;
    private boolean isError = false;

    public PlayModeItemModel(String str) {
        this.single = str;
    }

    public int getCount() {
        return this.count;
    }

    public int getEnd() {
        return this.end;
    }

    public int getEnd2() {
        return this.end2;
    }

    public int getFunData0() {
        return this.funData0;
    }

    public int getFunData1() {
        return this.funData1;
    }

    public int getMilli() {
        return this.milli;
    }

    public int getSeconds() {
        return this.seconds;
    }

    public int getSerialId() {
        return this.serialId;
    }

    public String getSingle() {
        return this.single;
    }

    public int getSmooth() {
        return this.smooth;
    }

    public int getStart() {
        return this.start;
    }

    public int getStart2() {
        return this.start2;
    }

    public boolean isError() {
        return this.isError;
    }

    public void setCount(int i) {
        this.count = i;
    }

    public void setEnd(int i) {
        this.end = i;
    }

    public void setEnd2(int i) {
        this.end2 = i;
    }

    public void setError(boolean z) {
        this.isError = z;
    }

    public void setFunData0(int i) {
        this.funData0 = i;
    }

    public void setFunData1(int i) {
        this.funData1 = i;
    }

    public void setMilli(int i) {
        this.milli = i;
    }

    public void setSeconds(int i) {
        this.seconds = i;
    }

    public void setSerialId(int i) {
        this.serialId = i;
    }

    public void setSingle(String str) {
        this.single = str;
    }

    public void setSmooth(int i) {
        this.smooth = i;
    }

    public void setStart(int i) {
        this.start = i;
    }

    public void setStart2(int i) {
        this.start2 = i;
    }

    public String toString() {
        return "PlayModeItemModel{serialId=" + this.serialId + ", single='" + this.single + "', start=" + this.start + ", end=" + this.end + ", start2=" + this.start2 + ", end2=" + this.end2 + ", seconds=" + this.seconds + ", milli=" + this.milli + ", count=" + this.count + ", smooth=" + this.smooth + ", funData0=" + this.funData0 + ", funData1=" + this.funData1 + ", isError=" + this.isError + '}';
    }
}
