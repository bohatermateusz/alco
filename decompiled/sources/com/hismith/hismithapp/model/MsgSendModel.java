package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class MsgSendModel extends HttpModel {
    private String content;
    private String date;
    private int firstChat;
    private String localId;
    private int msgId;
    private int referenceId;
    private String type;
    private int userId;
    private int voiceLength;

    public String getContent() {
        return this.content;
    }

    public String getDate() {
        return this.date;
    }

    public int getFirstChat() {
        return this.firstChat;
    }

    public String getLocalId() {
        return this.localId;
    }

    public int getMsgId() {
        return this.msgId;
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

    public int getVoiceLength() {
        return this.voiceLength;
    }

    public void setContent(String str) {
        this.content = str;
    }

    public void setDate(String str) {
        this.date = str;
    }

    public void setFirstChat(int i) {
        this.firstChat = i;
    }

    public void setLocalId(String str) {
        this.localId = str;
    }

    public void setMsgId(int i) {
        this.msgId = i;
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

    public void setVoiceLength(int i) {
        this.voiceLength = i;
    }

    public String toString() {
        return "MsgSendModel{msgId=" + this.msgId + ", localId='" + this.localId + "', userId=" + this.userId + ", referenceId=" + this.referenceId + ", content='" + this.content + "', voiceLength=" + this.voiceLength + ", type='" + this.type + "', firstChat=" + this.firstChat + ", date='" + this.date + "'}";
    }
}
