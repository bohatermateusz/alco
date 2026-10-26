package com.hismith.hismithapp.model.socket;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class PlayLineSocketModel extends HttpModel {
    private String action;
    private String clientHeadIcon;
    private int clientId;
    private String clientName;
    private int[] delRoom;
    private int[] delUser;
    private int home;
    private String message;
    private SocketRoomModel[] room;
    private SocketRoomModel[] rooms;
    private int strokeNow;
    private int strokeType;
    private SocketUserModel[] user;
    private String userHeadIcon;
    private int userId;
    private String userName;
    private int userNum;
    private int userRoomId;
    private SocketUserModel[] users;

    public String getAction() {
        return this.action;
    }

    public String getClientHeadIcon() {
        return this.clientHeadIcon;
    }

    public int getClientId() {
        return this.clientId;
    }

    public String getClientName() {
        return this.clientName;
    }

    public int[] getDelRoom() {
        return this.delRoom;
    }

    public int[] getDelUser() {
        return this.delUser;
    }

    public int getHome() {
        return this.home;
    }

    public String getMessage() {
        return this.message;
    }

    public SocketRoomModel[] getRoom() {
        return this.room;
    }

    public SocketRoomModel[] getRooms() {
        return this.rooms;
    }

    public int getStrokeNow() {
        return this.strokeNow;
    }

    public int getStrokeType() {
        return this.strokeType;
    }

    public SocketUserModel[] getUser() {
        return this.user;
    }

    public String getUserHeadIcon() {
        return this.userHeadIcon;
    }

    public int getUserId() {
        return this.userId;
    }

    public String getUserName() {
        return this.userName;
    }

    public int getUserNum() {
        return this.userNum;
    }

    public int getUserRoomId() {
        return this.userRoomId;
    }

    public SocketUserModel[] getUsers() {
        return this.users;
    }

    public void setAction(String str) {
        this.action = str;
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

    public void setDelRoom(int[] iArr) {
        this.delRoom = iArr;
    }

    public void setDelUser(int[] iArr) {
        this.delUser = iArr;
    }

    public void setHome(int i) {
        this.home = i;
    }

    public void setMessage(String str) {
        this.message = str;
    }

    public void setRoom(SocketRoomModel[] socketRoomModelArr) {
        this.room = socketRoomModelArr;
    }

    public void setRooms(SocketRoomModel[] socketRoomModelArr) {
        this.rooms = socketRoomModelArr;
    }

    public void setStrokeNow(int i) {
        this.strokeNow = i;
    }

    public void setStrokeType(int i) {
        this.strokeType = i;
    }

    public void setUser(SocketUserModel[] socketUserModelArr) {
        this.user = socketUserModelArr;
    }

    public void setUserHeadIcon(String str) {
        this.userHeadIcon = str;
    }

    public void setUserId(int i) {
        this.userId = i;
    }

    public void setUserName(String str) {
        this.userName = str;
    }

    public void setUserNum(int i) {
        this.userNum = i;
    }

    public void setUserRoomId(int i) {
        this.userRoomId = i;
    }

    public void setUsers(SocketUserModel[] socketUserModelArr) {
        this.users = socketUserModelArr;
    }
}
