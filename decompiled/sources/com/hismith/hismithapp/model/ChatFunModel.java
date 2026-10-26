package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class ChatFunModel {
    private int icon;
    private String name;
    private String type;

    public ChatFunModel(String str, String str2, int i) {
        this.name = str;
        this.type = str2;
        this.icon = i;
    }

    public int getIcon() {
        return this.icon;
    }

    public String getName() {
        return this.name;
    }

    public String getType() {
        return this.type;
    }

    public void setIcon(int i) {
        this.icon = i;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setType(String str) {
        this.type = str;
    }
}
