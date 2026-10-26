package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class ModeSettingModel {
    private int smooth;
    private int strokeMax;
    private int strokeMin;

    public ModeSettingModel() {
        this.strokeMin = 0;
        this.strokeMax = 100;
        this.smooth = 10;
    }

    public ModeSettingModel(int i, int i2, int i3) {
        this.strokeMin = i;
        this.strokeMax = i2;
        this.smooth = i3;
    }

    public int getSmooth() {
        return this.smooth;
    }

    public int getStrokeMax() {
        return this.strokeMax;
    }

    public int getStrokeMin() {
        return this.strokeMin;
    }

    public void setSmooth(int i) {
        this.smooth = i;
    }

    public void setStrokeMax(int i) {
        this.strokeMax = i;
    }

    public void setStrokeMin(int i) {
        this.strokeMin = i;
    }

    public String toString() {
        return "ModeSettingModel{strokeMin=" + this.strokeMin + ", strokeMax=" + this.strokeMax + ", smooth=" + this.smooth + '}';
    }
}
