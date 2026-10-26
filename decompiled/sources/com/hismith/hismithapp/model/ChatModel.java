package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class ChatModel extends HttpModel {
    private Integer age;
    private String constellation;
    private String content;
    private String date;
    private Integer gender;
    private String headIcon;
    private Boolean king;
    private Integer msgId;
    private String name;
    private Integer revoke;
    private String type;
    private Integer userId;
    private Integer voiceLength;

    public Integer getAge() {
        return this.age;
    }

    public String getConstellation() {
        return this.constellation;
    }

    public String getContent() {
        return this.content;
    }

    public String getDate() {
        return this.date;
    }

    public Integer getGender() {
        return this.gender;
    }

    public String getHeadIcon() {
        return this.headIcon;
    }

    public Boolean getKing() {
        return this.king;
    }

    public Integer getMsgId() {
        return this.msgId;
    }

    public String getName() {
        return this.name;
    }

    public Integer getRevoke() {
        return this.revoke;
    }

    public String getType() {
        return this.type;
    }

    public Integer getUserId() {
        return this.userId;
    }

    public Integer getVoiceLength() {
        return this.voiceLength;
    }

    public void setAge(Integer num) {
        this.age = num;
    }

    public void setConstellation(String str) {
        this.constellation = str;
    }

    public void setContent(String str) {
        this.content = str;
    }

    public void setDate(String str) {
        this.date = str;
    }

    public void setGender(Integer num) {
        this.gender = num;
    }

    public void setHeadIcon(String str) {
        this.headIcon = str;
    }

    public void setKing(Boolean bool) {
        this.king = bool;
    }

    public void setMsgId(Integer num) {
        this.msgId = num;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setRevoke(Integer num) {
        this.revoke = num;
    }

    public void setType(String str) {
        this.type = str;
    }

    public void setUserId(Integer num) {
        this.userId = num;
    }

    public void setVoiceLength(Integer num) {
        this.voiceLength = num;
    }
}
