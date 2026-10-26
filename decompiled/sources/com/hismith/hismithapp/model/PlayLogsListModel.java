package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class PlayLogsListModel extends HttpModel {
    private PlayLogsModel[] playLogs;
    private String requestId;
    private int totalCount;

    public PlayLogsModel[] getPlayLogs() {
        return this.playLogs;
    }

    public String getRequestId() {
        return this.requestId;
    }

    public int getTotalCount() {
        return this.totalCount;
    }

    public void setPlayLogs(PlayLogsModel[] playLogsModelArr) {
        this.playLogs = playLogsModelArr;
    }

    public void setRequestId(String str) {
        this.requestId = str;
    }

    public void setTotalCount(int i) {
        this.totalCount = i;
    }

    public String toString() {
        return "PlayLogsListModel{totalCount=" + this.totalCount + ", requestId='" + this.requestId + "', playLogs=" + Arrays.toString(this.playLogs) + '}';
    }
}
