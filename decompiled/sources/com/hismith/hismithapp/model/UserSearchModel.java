package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class UserSearchModel extends HttpModel {
    private int page;
    private int pageSize;
    private int totalCount;
    private int totalPage;
    private User[] users;

    public class User {
        private String birthday;
        private int gender;
        private String gradeIcon;
        private String headIcon;
        private int isFollow;
        private int king;
        private String name;
        private int online;
        private int userId;

        public User() {
        }

        public String getBirthday() {
            return this.birthday;
        }

        public int getGender() {
            return this.gender;
        }

        public String getGradeIcon() {
            return this.gradeIcon;
        }

        public String getHeadIcon() {
            return this.headIcon;
        }

        public int getIsFollow() {
            return this.isFollow;
        }

        public int getKing() {
            return this.king;
        }

        public String getName() {
            return this.name;
        }

        public int getOnline() {
            return this.online;
        }

        public int getUserId() {
            return this.userId;
        }

        public void setBirthday(String str) {
            this.birthday = str;
        }

        public void setGender(int i) {
            this.gender = i;
        }

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
        }

        public void setHeadIcon(String str) {
            this.headIcon = str;
        }

        public void setIsFollow(int i) {
            this.isFollow = i;
        }

        public void setKing(int i) {
            this.king = i;
        }

        public void setName(String str) {
            this.name = str;
        }

        public void setOnline(int i) {
            this.online = i;
        }

        public void setUserId(int i) {
            this.userId = i;
        }

        public String toString() {
            return "User{userId=" + this.userId + ", king=" + this.king + ", gradeIcon='" + this.gradeIcon + "', name='" + this.name + "', gender=" + this.gender + ", online=" + this.online + ", birthday='" + this.birthday + "', headIcon='" + this.headIcon + "', isFollow=" + this.isFollow + '}';
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
        return "UserSearchModel{users=" + Arrays.toString(this.users) + ", page=" + this.page + ", pageSize=" + this.pageSize + ", totalCount=" + this.totalCount + ", totalPage=" + this.totalPage + '}';
    }
}
