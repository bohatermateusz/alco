package com.hismith.hismithapp.model;

import java.io.Serializable;

/* loaded from: classes3.dex */
public class UserPhotosModel implements Serializable {
    private int sex;
    private String url;

    public int getSex() {
        return this.sex;
    }

    public String getUrl() {
        return this.url;
    }

    public void setSex(int i) {
        this.sex = i;
    }

    public void setUrl(String str) {
        this.url = str;
    }

    public String toString() {
        return "UserPhotosModel{url='" + this.url + "', sex=" + this.sex + '}';
    }
}
