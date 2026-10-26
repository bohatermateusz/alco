package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class TopicSendModel extends HttpModel {
    private Integer id;
    private String localId;

    public Integer getId() {
        return this.id;
    }

    public String getLocalId() {
        return this.localId;
    }

    public void setId(Integer num) {
        this.id = num;
    }

    public void setLocalId(String str) {
        this.localId = str;
    }

    public String toString() {
        return "TopicSendModel{id=" + this.id + ", localId='" + this.localId + "'}";
    }
}
