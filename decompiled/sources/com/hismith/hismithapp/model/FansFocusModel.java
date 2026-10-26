package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class FansFocusModel extends HttpModel {
    private int next;
    private int page;
    private int pageSize;
    private int totalCount;
    private int totalPage;
    private User[] users;

    public class User {
        private String birthday;
        private int followStatus;
        private Integer gender;
        private String gradeIcon;
        private String headIcon;
        private Integer isField;
        private int isFollow = 1;
        private Integer king;
        private String name;
        private int online;
        private int userId;

        public User() {
        }

        public int IsFollow() {
            return this.isFollow;
        }

        public String getBirthday() {
            return this.birthday;
        }

        public int getFollowStatus() {
            return this.followStatus;
        }

        public Integer getGender() {
            return this.gender;
        }

        public String getGradeIcon() {
            return this.gradeIcon;
        }

        public String getHeadIcon() {
            return this.headIcon;
        }

        public Integer getIsField() {
            return this.isField;
        }

        public Integer getKing() {
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

        public void setFollow(int i) {
            this.isFollow = i;
        }

        public void setFollowStatus(int i) {
            this.followStatus = i;
        }

        public void setGender(Integer num) {
            this.gender = num;
        }

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
        }

        public void setHeadIcon(String str) {
            this.headIcon = str;
        }

        public void setIsField(Integer num) {
            this.isField = num;
        }

        public void setKing(Integer num) {
            this.king = num;
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
            return "User{userId=" + this.userId + ", king=" + this.king + ", gradeIcon='" + this.gradeIcon + "', name='" + this.name + "', gender=" + this.gender + ", online=" + this.online + ", isField=" + this.isField + ", birthday='" + this.birthday + "', headIcon='" + this.headIcon + "', isFollow=" + this.isFollow + ", followStatus=" + this.followStatus + '}';
        }
    }

    public int getNext() {
        return this.next;
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

    public void setNext(int i) {
        this.next = i;
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
        return "FansFocusModel{users=" + Arrays.toString(this.users) + ", next=" + this.next + ", page=" + this.page + ", pageSize=" + this.pageSize + ", totalCount=" + this.totalCount + ", totalPage=" + this.totalPage + '}';
    }
}
