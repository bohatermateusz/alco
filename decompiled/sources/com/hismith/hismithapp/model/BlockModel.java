package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class BlockModel extends HttpModel {
    private int isBlocked;
    private String userId;

    public int getIsBlocked() {
        return this.isBlocked;
    }

    public String getUserId() {
        return this.userId;
    }

    public void setIsBlocked(int i) {
        this.isBlocked = i;
    }

    public void setUserId(String str) {
        this.userId = str;
    }

    public String toString() {
        return "BlockModel{userId='" + this.userId + "', isBlocked=" + this.isBlocked + '}';
    }
}
