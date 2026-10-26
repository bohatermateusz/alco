package com.hismith.hismithapp.model.lomi;

import com.hismith.hismithapp.model.MessageVoiceModel;

/* loaded from: classes3.dex */
public class LomiMessage {
    private int androidSend;
    private int status;
    private String text;
    private long timestamp;
    private String type;
    private MessageVoiceModel voice;

    public int getAndroidSend() {
        return this.androidSend;
    }

    public int getStatus() {
        return this.status;
    }

    public String getText() {
        return this.text;
    }

    public long getTimestamp() {
        return this.timestamp;
    }

    public String getType() {
        return this.type;
    }

    public MessageVoiceModel getVoice() {
        return this.voice;
    }

    public void setAndroidSend(int i) {
        this.androidSend = i;
    }

    public void setStatus(int i) {
        this.status = i;
    }

    public void setText(String str) {
        this.text = str;
    }

    public void setTimestamp(long j) {
        this.timestamp = j;
    }

    public void setType(String str) {
        this.type = str;
    }

    public void setVoice(MessageVoiceModel messageVoiceModel) {
        this.voice = messageVoiceModel;
    }

    public String toString() {
        return "Message{type='" + this.type + "', text='" + this.text + "', voice=" + this.voice + ", androidSend=" + this.androidSend + ", status=" + this.status + ", timestamp=" + this.timestamp + '}';
    }
}
