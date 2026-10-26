package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class LineRoomInfoModel extends HttpModel {
    private int isFollow;
    private Play play;
    private RoomInfoUserModel[] user;
    private int userCount;
    private String userHeadIcon;
    private String userId;
    private String userName;

    public class Play {
        private int mode;
        private String modeData;
        private String modePoints;
        private int modeStop;
        private String modeTime;
        private float playTime;
        private int speed;
        private int strokeType;
        private int vspeed;

        public Play() {
        }

        public int getMode() {
            return this.mode;
        }

        public String getModeData() {
            return this.modeData;
        }

        public String getModePoints() {
            return this.modePoints;
        }

        public int getModeStop() {
            return this.modeStop;
        }

        public String getModeTime() {
            return this.modeTime;
        }

        public float getPlayTime() {
            return this.playTime;
        }

        public int getSpeed() {
            return this.speed;
        }

        public int getStrokeType() {
            return this.strokeType;
        }

        public int getVspeed() {
            return this.vspeed;
        }

        public void setMode(int i) {
            this.mode = i;
        }

        public void setModeData(String str) {
            this.modeData = str;
        }

        public void setModePoints(String str) {
            this.modePoints = str;
        }

        public void setModeStop(int i) {
            this.modeStop = i;
        }

        public void setModeTime(String str) {
            this.modeTime = str;
        }

        public void setPlayTime(float f) {
            this.playTime = f;
        }

        public void setSpeed(int i) {
            this.speed = i;
        }

        public void setStrokeType(int i) {
            this.strokeType = i;
        }

        public void setVspeed(int i) {
            this.vspeed = i;
        }
    }

    public class RoomInfoUserModel {
        private int captain;
        private String userHeadIcon;
        private String userId;
        private String userName;

        public RoomInfoUserModel() {
        }

        public int getCaptain() {
            return this.captain;
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

        public void setCaptain(int i) {
            this.captain = i;
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
            return "RoomInfoUserModel{captain=" + this.captain + ", userId='" + this.userId + "', userName='" + this.userName + "', userHeadIcon='" + this.userHeadIcon + "'}";
        }
    }

    public int getIsFollow() {
        return this.isFollow;
    }

    public Play getPlay() {
        return this.play;
    }

    public RoomInfoUserModel[] getUser() {
        return this.user;
    }

    public int getUserCount() {
        return this.userCount;
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

    public void setIsFollow(int i) {
        this.isFollow = i;
    }

    public void setPlay(Play play) {
        this.play = play;
    }

    public void setUser(RoomInfoUserModel[] roomInfoUserModelArr) {
        this.user = roomInfoUserModelArr;
    }

    public void setUserCount(int i) {
        this.userCount = i;
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
        return "LineRoomInfoModel{userId='" + this.userId + "', userName='" + this.userName + "', userHeadIcon='" + this.userHeadIcon + "', userCount=" + this.userCount + ", isFollow=" + this.isFollow + ", user=" + Arrays.toString(this.user) + '}';
    }
}
