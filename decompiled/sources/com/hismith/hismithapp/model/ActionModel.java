package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class ActionModel {
    private boolean isNew = false;
    private int logoInt;
    private String name;

    public ActionModel(int i, String str) {
        this.logoInt = i;
        this.name = str;
    }

    public int getLogoInt() {
        return this.logoInt;
    }

    public String getName() {
        return this.name;
    }

    public boolean isNew() {
        return this.isNew;
    }

    public void setLogoInt(int i) {
        this.logoInt = i;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setNew(boolean z) {
        this.isNew = z;
    }
}
