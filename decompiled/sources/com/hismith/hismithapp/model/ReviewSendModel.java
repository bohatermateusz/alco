package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class ReviewSendModel extends HttpModel {
    private Integer id;
    private Integer rid;

    public Integer getId() {
        return this.id;
    }

    public Integer getRid() {
        return this.rid;
    }

    public void setId(Integer num) {
        this.id = num;
    }

    public void setRid(Integer num) {
        this.rid = num;
    }

    public String toString() {
        return "ReviewSendModel{id=" + this.id + ", rid=" + this.rid + '}';
    }
}
