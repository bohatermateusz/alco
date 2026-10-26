package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class ReferenceMsg {
    private String content;
    private String localPath;
    private int referenceId;
    private String type;
    private int userId;
    private String userName;
    private int voiceLength;

    public String getContent() {
        return this.content;
    }

    public String getLocalPath() {
        return this.localPath;
    }

    public int getReferenceId() {
        return this.referenceId;
    }

    public String getType() {
        return this.type;
    }

    public int getUserId() {
        return this.userId;
    }

    public String getUserName() {
        return this.userName;
    }

    public int getVoiceLength() {
        return this.voiceLength;
    }

    public void setContent(String str) {
        this.content = str;
    }

    public void setLocalPath(String str) {
        this.localPath = str;
    }

    public void setReferenceId(int i) {
        this.referenceId = i;
    }

    public void setType(String str) {
        this.type = str;
    }

    public void setUserId(int i) {
        this.userId = i;
    }

    public void setUserName(String str) {
        this.userName = str;
    }

    public void setVoiceLength(int i) {
        this.voiceLength = i;
    }

    public String toString() {
        return "ReferenceMsg{referenceId=" + this.referenceId + ", type='" + this.type + "', voiceLength=" + this.voiceLength + ", content='" + this.content + "', userId=" + this.userId + ", userName='" + this.userName + "', localPath='" + this.localPath + "'}";
    }
}
