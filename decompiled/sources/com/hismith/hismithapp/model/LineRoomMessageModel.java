package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class LineRoomMessageModel extends HttpModel {
    private String clientHeadIcon;
    private int clientId;
    private String clientName;
    private String message;
    private String userHeadIcon;
    private String userId;
    private String userName;

    public String getClientHeadIcon() {
        return this.clientHeadIcon;
    }

    public int getClientId() {
        return this.clientId;
    }

    public String getClientName() {
        return this.clientName;
    }

    public String getMessage() {
        return this.message;
    }

    public String getUserHeadIcon() {
        return this.userHeadIcon;
    }

    public String getUserId() {
        return this.userId;
    }

    public String getUserName() {
        return this.userName;
    }

    public void setClientHeadIcon(String str) {
        this.clientHeadIcon = str;
    }

    public void setClientId(int i) {
        this.clientId = i;
    }

    public void setClientName(String str) {
        this.clientName = str;
    }

    public void setMessage(String str) {
        this.message = str;
    }

    public void setUserHeadIcon(String str) {
        this.userHeadIcon = str;
    }

    public void setUserId(String str) {
        this.userId = str;
    }

    public void setUserName(String str) {
        this.userName = str;
    }

    public String toString() {
        return "LineRoomMessageModel{userId='" + this.userId + "', userName='" + this.userName + "', userHeadIcon='" + this.userHeadIcon + "', message='" + this.message + "', clientId=" + this.clientId + ", clientName='" + this.clientName + "', clientHeadIcon='" + this.clientHeadIcon + "'}";
    }
}
