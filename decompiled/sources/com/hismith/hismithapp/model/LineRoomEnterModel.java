package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class LineRoomEnterModel {
    private int isFollow;
    private int isPublic;
    private int roomId;
    private String userId;
    private String userName;
    private String welcomeMsg;

    public int getIsFollow() {
        return this.isFollow;
    }

    public int getIsPublic() {
        return this.isPublic;
    }

    public int getRoomId() {
        return this.roomId;
    }

    public String getUserId() {
        return this.userId;
    }

    public String getUserName() {
        return this.userName;
    }

    public String getWelcomeMsg() {
        return this.welcomeMsg;
    }

    public void setIsFollow(int i) {
        this.isFollow = i;
    }

    public void setIsPublic(int i) {
        this.isPublic = i;
    }

    public void setRoomId(int i) {
        this.roomId = i;
    }

    public void setUserId(String str) {
        this.userId = str;
    }

    public void setUserName(String str) {
        this.userName = str;
    }

    public void setWelcomeMsg(String str) {
        this.welcomeMsg = str;
    }

    public String toString() {
        return "LineRoomEnterModel{roomId=" + this.roomId + ", isPublic=" + this.isPublic + ", userId='" + this.userId + "', userName='" + this.userName + "', welcomeMsg='" + this.welcomeMsg + "', isFollow=" + this.isFollow + '}';
    }
}
