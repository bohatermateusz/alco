package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class PlayModeListModel extends HttpModel {
    private PlayModeModel[] playModes;
    private String requestId;
    private String totalCount;

    public PlayModeModel[] getPlayModes() {
        return this.playModes;
    }

    public String getRequestId() {
        return this.requestId;
    }

    public String getTotalCount() {
        return this.totalCount;
    }

    public void setPlayModes(PlayModeModel[] playModeModelArr) {
        this.playModes = playModeModelArr;
    }

    public void setRequestId(String str) {
        this.requestId = str;
    }

    public void setTotalCount(String str) {
        this.totalCount = str;
    }

    public String toString() {
        return "PlayModeListModel{playModes=" + Arrays.toString(this.playModes) + ", requestId='" + this.requestId + "', totalCount='" + this.totalCount + "'}";
    }
}
