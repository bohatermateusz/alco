package com.hismith.hismithapp.model.socket;

/* loaded from: classes3.dex */
public class SocketRoomModel {
    private int isPublic;
    private int roomId;
    private String roomName;
    private int roomPlay = -1;
    private int userCount;
    private int userGender;
    private String userHeadIcon;
    private int userId;
    private int userLimit;
    private String userName;

    public int getIsPublic() {
        return this.isPublic;
    }

    public int getRoomId() {
        return this.roomId;
    }

    public String getRoomName() {
        return this.roomName;
    }

    public int getRoomPlay() {
        return this.roomPlay;
    }

    public int getUserCount() {
        return this.userCount;
    }

    public int getUserGender() {
        return this.userGender;
    }

    public String getUserHeadIcon() {
        return this.userHeadIcon;
    }

    public int getUserId() {
        return this.userId;
    }

    public int getUserLimit() {
        return this.userLimit;
    }

    public String getUserName() {
        return this.userName;
    }

    public void setIsPublic(int i) {
        this.isPublic = i;
    }

    public void setRoomId(int i) {
        this.roomId = i;
    }

    public void setRoomName(String str) {
        this.roomName = str;
    }

    public void setRoomPlay(int i) {
        this.roomPlay = i;
    }

    public void setUserCount(int i) {
        this.userCount = i;
    }

    public void setUserGender(int i) {
        this.userGender = i;
    }

    public void setUserHeadIcon(String str) {
        this.userHeadIcon = str;
    }

    public void setUserId(int i) {
        this.userId = i;
    }

    public void setUserLimit(int i) {
        this.userLimit = i;
    }

    public void setUserName(String str) {
        this.userName = str;
    }

    public String toString() {
        return "SocketRoomModel{roomId=" + this.roomId + ", roomName='" + this.roomName + "', roomPlay=" + this.roomPlay + ", isPublic=" + this.isPublic + ", userId=" + this.userId + ", userName='" + this.userName + "', userGender=" + this.userGender + ", userHeadIcon='" + this.userHeadIcon + "', userLimit=" + this.userLimit + ", userCount=" + this.userCount + '}';
    }
}
