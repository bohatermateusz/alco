package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class PlaySelectorModel {
    private int gradientRes;
    private int iconRes;
    private String title;

    public PlaySelectorModel(String str, int i, int i2) {
        this.title = str;
        this.iconRes = i;
        this.gradientRes = i2;
    }

    public int getGradientRes() {
        return this.gradientRes;
    }

    public int getIconRes() {
        return this.iconRes;
    }

    public String getTitle() {
        return this.title;
    }
}
