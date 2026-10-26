package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class TopicActionModel extends HttpModel {
    private String collectCount;
    private String likeCount;

    public String getCollectCount() {
        return this.collectCount;
    }

    public String getLikeCount() {
        return this.likeCount;
    }

    public void setCollectCount(String str) {
        this.collectCount = str;
    }

    public void setLikeCount(String str) {
        this.likeCount = str;
    }

    public String toString() {
        return "TopicActionModel{likeCount=" + this.likeCount + ", collectCount=" + this.collectCount + '}';
    }
}
