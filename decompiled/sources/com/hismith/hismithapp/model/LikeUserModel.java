package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class LikeUserModel extends HttpModel {
    private int page;
    private int pageSize;
    private int totalCount;
    private int totalPage;
    private User[] users;

    public class User {
        private String gradeIcon;
        private int userGender;
        private String userHeadIcon;
        private int userId;
        private int userKing;
        private String userName;

        public User() {
        }

        public String getGradeIcon() {
            return this.gradeIcon;
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

        public int getUserKing() {
            return this.userKing;
        }

        public String getUserName() {
            return this.userName;
        }

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
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

        public void setUserKing(int i) {
            this.userKing = i;
        }

        public void setUserName(String str) {
            this.userName = str;
        }

        public String toString() {
            return "User{userId=" + this.userId + ", userKing=" + this.userKing + ", userName='" + this.userName + "', userGender=" + this.userGender + ", userHeadIcon='" + this.userHeadIcon + "'}";
        }
    }

    public int getPage() {
        return this.page;
    }

    public int getPageSize() {
        return this.pageSize;
    }

    public int getTotalCount() {
        return this.totalCount;
    }

    public int getTotalPage() {
        return this.totalPage;
    }

    public User[] getUsers() {
        return this.users;
    }

    public void setPage(int i) {
        this.page = i;
    }

    public void setPageSize(int i) {
        this.pageSize = i;
    }

    public void setTotalCount(int i) {
        this.totalCount = i;
    }

    public void setTotalPage(int i) {
        this.totalPage = i;
    }

    public void setUsers(User[] userArr) {
        this.users = userArr;
    }

    public String toString() {
        return "LikeUserModel{users=" + Arrays.toString(this.users) + ", page=" + this.page + ", pageSize=" + this.pageSize + ", totalCount=" + this.totalCount + ", totalPage=" + this.totalPage + '}';
    }
}
