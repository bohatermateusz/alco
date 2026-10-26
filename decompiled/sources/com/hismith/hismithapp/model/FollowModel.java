package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class FollowModel extends HttpModel {
    private Integer follow;
    private Integer userId;

    public Integer getFollow() {
        return this.follow;
    }

    public Integer getUserId() {
        return this.userId;
    }

    public void setFollow(Integer num) {
        this.follow = num;
    }

    public void setUserId(Integer num) {
        this.userId = num;
    }

    public String toString() {
        return "FollowModel{userId=" + this.userId + ", follow=" + this.follow + '}';
    }
}
