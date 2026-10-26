package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class FunDatModel {
    private int funId;
    private int mode;
    private int speed;

    public int getFunId() {
        return this.funId;
    }

    public int getMode() {
        return this.mode;
    }

    public int getSpeed() {
        return this.speed;
    }

    public void setFunId(int i) {
        this.funId = i;
    }

    public void setMode(int i) {
        this.mode = i;
    }

    public void setSpeed(int i) {
        this.speed = i;
    }

    public String toString() {
        return "FunDatModel{funId=" + this.funId + ", speed=" + this.speed + ", mode=" + this.mode + '}';
    }
}
