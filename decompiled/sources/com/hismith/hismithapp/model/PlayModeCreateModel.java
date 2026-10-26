package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class PlayModeCreateModel extends HttpModel {
    private int id;
    private String modeData;
    private String modePoints;
    private String modeSetting;

    public int getId() {
        return this.id;
    }

    public String getModeData() {
        return this.modeData;
    }

    public String getModePoints() {
        return this.modePoints;
    }

    public String getModeSetting() {
        return this.modeSetting;
    }

    public void setId(int i) {
        this.id = i;
    }

    public void setModeData(String str) {
        this.modeData = str;
    }

    public void setModePoints(String str) {
        this.modePoints = str;
    }

    public void setModeSetting(String str) {
        this.modeSetting = str;
    }

    public String toString() {
        return "PlayModeCreateModel{id=" + this.id + ", modeData='" + this.modeData + "', modePoints='" + this.modePoints + "', modeSetting='" + this.modeSetting + "'}";
    }
}
