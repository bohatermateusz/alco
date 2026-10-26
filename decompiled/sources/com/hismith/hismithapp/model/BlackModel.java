package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class BlackModel extends HttpModel {
    private int next;
    private int page;
    private int pageSize;
    private int totalCount;
    private int totalPage;
    private User[] users;

    public class User {
        private String addDate;
        private int checked;
        private int gender;
        private String gradeIcon;
        private String headIcon;
        private int isField;
        private int king;
        private String name;
        private int online;
        private int userId;

        public User() {
        }

        public String getAddDate() {
            return this.addDate;
        }

        public int getChecked() {
            return this.checked;
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

        public int getIsField() {
            return this.isField;
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

        public void setAddDate(String str) {
            this.addDate = str;
        }

        public void setChecked(int i) {
            this.checked = i;
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

        public void setIsField(int i) {
            this.isField = i;
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
            return "User{userId=" + this.userId + ", king=" + this.king + ", gradeIcon='" + this.gradeIcon + "', name='" + this.name + "', gender=" + this.gender + ", online=" + this.online + ", isField=" + this.isField + ", headIcon='" + this.headIcon + "', addDate='" + this.addDate + "', checked=" + this.checked + '}';
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
        return "BlackModel{users=" + Arrays.toString(this.users) + ", next=" + this.next + ", page=" + this.page + ", pageSize=" + this.pageSize + ", totalCount=" + this.totalCount + ", totalPage=" + this.totalPage + '}';
    }
}
