package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class UploadModel extends HttpModel {
    private String fileUrl;
    private String requestId;

    public String getFileUrl() {
        return this.fileUrl;
    }

    public String getRequestId() {
        return this.requestId;
    }

    public void setFileUrl(String str) {
        this.fileUrl = str;
    }

    public void setRequestId(String str) {
        this.requestId = str;
    }

    public String toString() {
        return "UploadModel{requestId='" + this.requestId + "', fileUrl='" + this.fileUrl + "'}";
    }
}
