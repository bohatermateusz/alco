package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class DanMuModel extends HttpModel implements Comparable<DanMuModel> {
    private String content;
    private int time;

    public DanMuModel(String str, int i) {
        this.content = str;
        this.time = i;
    }

    @Override // java.lang.Comparable
    public int compareTo(DanMuModel danMuModel) {
        if (getTime() < danMuModel.getTime()) {
            return -1;
        }
        return getTime() > danMuModel.getTime() ? 1 : 0;
    }

    public String getContent() {
        return this.content;
    }

    public int getTime() {
        return this.time;
    }

    public void setContent(String str) {
        this.content = str;
    }

    public void setTime(int i) {
        this.time = i;
    }

    public String toString() {
        return "DanMuModel{content='" + this.content + "', time=" + this.time + '}';
    }
}
