package com.hismith.hismithapp.model;

import com.google.gson.JsonElement;
import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class ComplaintReasonsModel extends HttpModel {
    private JsonElement complaintReasons;

    public JsonElement getComplaintReasons() {
        return this.complaintReasons;
    }

    public void setComplaintReasons(JsonElement jsonElement) {
        this.complaintReasons = jsonElement;
    }

    public String toString() {
        return "ComplaintReasonsModel{complaintReasons=" + this.complaintReasons + '}';
    }
}
