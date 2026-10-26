package com.hismith.hismithapp.manager;

/* loaded from: classes3.dex */
public class DownloadInfo {
    public static final long TOTAL_ERROR = -1;
    private String fileName;
    private long progress;
    private long total;
    private String url;

    public DownloadInfo(String str) {
        this.url = str;
    }

    public String getFileName() {
        return this.fileName;
    }

    public long getProgress() {
        return this.progress;
    }

    public long getTotal() {
        return this.total;
    }

    public String getUrl() {
        return this.url;
    }

    public void setFileName(String str) {
        this.fileName = str;
    }

    public void setProgress(long j) {
        this.progress = j;
    }

    public void setTotal(long j) {
        this.total = j;
    }

    public void setUrl(String str) {
        this.url = str;
    }
}
