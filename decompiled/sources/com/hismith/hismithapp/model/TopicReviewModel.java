package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class TopicReviewModel extends HttpModel {
    private Integer page;
    private Integer pageSize;
    private Review[] reviews;
    private Integer totalCount;
    private Integer totalPage;

    public class Review {
        private String content;
        private String createDate;
        private String gradeIcon;
        private Integer id;
        private String image;
        private Integer likeCount;
        private Integer likeed;
        private Integer replyCount;
        private Integer userGender;
        private String userHeadIcon;
        private Integer userId;
        private Integer userKing;
        private String userName;

        public Review() {
        }

        public String getContent() {
            return this.content;
        }

        public String getCreateDate() {
            return this.createDate;
        }

        public String getGradeIcon() {
            return this.gradeIcon;
        }

        public Integer getId() {
            return this.id;
        }

        public String getImage() {
            return this.image;
        }

        public Integer getLikeCount() {
            return this.likeCount;
        }

        public Integer getLikeed() {
            return this.likeed;
        }

        public Integer getReplyCount() {
            return this.replyCount;
        }

        public Integer getUserGender() {
            return this.userGender;
        }

        public String getUserHeadIcon() {
            return this.userHeadIcon;
        }

        public Integer getUserId() {
            return this.userId;
        }

        public Integer getUserKing() {
            return this.userKing;
        }

        public String getUserName() {
            return this.userName;
        }

        public void setContent(String str) {
            this.content = str;
        }

        public void setCreateDate(String str) {
            this.createDate = str;
        }

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
        }

        public void setId(Integer num) {
            this.id = num;
        }

        public void setImage(String str) {
            this.image = str;
        }

        public void setLikeCount(Integer num) {
            this.likeCount = num;
        }

        public void setLikeed(Integer num) {
            this.likeed = num;
        }

        public void setReplyCount(Integer num) {
            this.replyCount = num;
        }

        public void setUserGender(Integer num) {
            this.userGender = num;
        }

        public void setUserHeadIcon(String str) {
            this.userHeadIcon = str;
        }

        public void setUserId(Integer num) {
            this.userId = num;
        }

        public void setUserKing(Integer num) {
            this.userKing = num;
        }

        public void setUserName(String str) {
            this.userName = str;
        }

        public String toString() {
            return "Review{id=" + this.id + ", image='" + this.image + "', content='" + this.content + "', createDate='" + this.createDate + "', likeCount=" + this.likeCount + ", likeed=" + this.likeed + ", replyCount=" + this.replyCount + ", userId=" + this.userId + ", userKing=" + this.userKing + ", gradeIcon='" + this.gradeIcon + "', userName='" + this.userName + "', userGender=" + this.userGender + ", userHeadIcon='" + this.userHeadIcon + "'}";
        }
    }

    public Integer getPage() {
        return this.page;
    }

    public Integer getPageSize() {
        return this.pageSize;
    }

    public Review[] getReviews() {
        return this.reviews;
    }

    public Integer getTotalCount() {
        return this.totalCount;
    }

    public Integer getTotalPage() {
        return this.totalPage;
    }

    public void setPage(Integer num) {
        this.page = num;
    }

    public void setPageSize(Integer num) {
        this.pageSize = num;
    }

    public void setReviews(Review[] reviewArr) {
        this.reviews = reviewArr;
    }

    public void setTotalCount(Integer num) {
        this.totalCount = num;
    }

    public void setTotalPage(Integer num) {
        this.totalPage = num;
    }

    public String toString() {
        return "TopicReviewModel{reviews=" + Arrays.toString(this.reviews) + ", page=" + this.page + ", pageSize=" + this.pageSize + ", totalCount=" + this.totalCount + ", totalPage=" + this.totalPage + '}';
    }
}
