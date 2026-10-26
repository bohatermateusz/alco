package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class ReviewLikeModel extends HttpModel {
    private Integer id;
    private Integer likeCount;

    public Integer getId() {
        return this.id;
    }

    public Integer getLikeCount() {
        return this.likeCount;
    }

    public void setId(Integer num) {
        this.id = num;
    }

    public void setLikeCount(Integer num) {
        this.likeCount = num;
    }

    public String toString() {
        return "ReviewLikeModel{id=" + this.id + ", likeCount=" + this.likeCount + '}';
    }
}
