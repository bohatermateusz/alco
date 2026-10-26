package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class ShareModel {
    private int logoInt;
    private String name;
    private int shareId;

    public ShareModel(int i, String str, int i2) {
        this.logoInt = i;
        this.name = str;
        this.shareId = i2;
    }

    public int getLogoInt() {
        return this.logoInt;
    }

    public String getName() {
        return this.name;
    }

    public int getShareId() {
        return this.shareId;
    }

    public void setLogoInt(int i) {
        this.logoInt = i;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setShareId(int i) {
        this.shareId = i;
    }
}
