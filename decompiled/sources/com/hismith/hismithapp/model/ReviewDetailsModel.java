package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class ReviewDetailsModel extends HttpModel {
    private String content;
    private String createDate;
    private String gradeIcon;
    private Integer id;
    private String image;
    private Integer likeCount;
    private String likeNote;
    private User[] likeUser;
    private Integer likeed;
    private Integer page;
    private Integer pageSize;
    private Integer replyCount;
    private Review[] replys;
    private Integer topicId;
    private String topicImage;
    private String topicTitle;
    private Integer totalCount;
    private Integer totalPage;
    private Integer userGender;
    private String userHeadIcon;
    private int userId;
    private Integer userKing;
    private String userName;

    public class Reply {
        private String content;
        private String gradeIcon;
        private Integer id;
        private String image;
        private Integer userGender;
        private String userHeadIcon;
        private Integer userId;
        private Integer userKing;
        private String userName;

        public Reply() {
        }

        public String getContent() {
            return this.content;
        }

        public Integer getId() {
            return this.id;
        }

        public String getImage() {
            return this.image;
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

        public void setId(Integer num) {
            this.id = num;
        }

        public void setImage(String str) {
            this.image = str;
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
    }

    public class Review {
        private String content;
        private String createDate;
        private String gradeIcon;
        private Integer id;
        private String image;
        private Integer likeCount;
        private Integer likeed;
        private Reply reply;
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

        public Reply getReply() {
            return this.reply;
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

        public void setReply(Reply reply) {
            this.reply = reply;
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
    }

    public class User {
        private String gradeIcon;
        private Integer userGender;
        private String userHeadIcon;
        private Integer userId;
        private Integer userKing;
        private String userName;

        public User() {
        }

        public String getGradeIcon() {
            return this.gradeIcon;
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

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
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

    public String getLikeNote() {
        return this.likeNote;
    }

    public User[] getLikeUser() {
        return this.likeUser;
    }

    public Integer getLikeed() {
        return this.likeed;
    }

    public Integer getPage() {
        return this.page;
    }

    public Integer getPageSize() {
        return this.pageSize;
    }

    public Integer getReplyCount() {
        return this.replyCount;
    }

    public Review[] getReplys() {
        return this.replys;
    }

    public Integer getTopicId() {
        return this.topicId;
    }

    public String getTopicImage() {
        return this.topicImage;
    }

    public String getTopicTitle() {
        return this.topicTitle;
    }

    public Integer getTotalCount() {
        return this.totalCount;
    }

    public Integer getTotalPage() {
        return this.totalPage;
    }

    public Integer getUserGender() {
        return this.userGender;
    }

    public String getUserHeadIcon() {
        return this.userHeadIcon;
    }

    public int getUserId() {
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

    public void setLikeNote(String str) {
        this.likeNote = str;
    }

    public void setLikeUser(User[] userArr) {
        this.likeUser = userArr;
    }

    public void setLikeed(Integer num) {
        this.likeed = num;
    }

    public void setPage(Integer num) {
        this.page = num;
    }

    public void setPageSize(Integer num) {
        this.pageSize = num;
    }

    public void setReplyCount(Integer num) {
        this.replyCount = num;
    }

    public void setReplys(Review[] reviewArr) {
        this.replys = reviewArr;
    }

    public void setTopicId(Integer num) {
        this.topicId = num;
    }

    public void setTopicImage(String str) {
        this.topicImage = str;
    }

    public void setTopicTitle(String str) {
        this.topicTitle = str;
    }

    public void setTotalCount(Integer num) {
        this.totalCount = num;
    }

    public void setTotalPage(Integer num) {
        this.totalPage = num;
    }

    public void setUserGender(Integer num) {
        this.userGender = num;
    }

    public void setUserHeadIcon(String str) {
        this.userHeadIcon = str;
    }

    public void setUserId(int i) {
        this.userId = i;
    }

    public void setUserKing(Integer num) {
        this.userKing = num;
    }

    public void setUserName(String str) {
        this.userName = str;
    }
}
