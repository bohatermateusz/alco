package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.io.Serializable;
import java.util.ArrayList;

/* loaded from: classes3.dex */
public class TopicModel extends HttpModel implements Serializable {
    private String collectCount;
    private String collectDate;
    private int collectStatus;
    private Integer collected;
    private String content;
    private String createDate;
    private String gradeIcon;
    private int hasSex;
    private Integer id;
    private ArrayList<String> images;
    private int isFans;
    private String likeCount;
    private int likeStatus;
    private Integer likeed;
    private String localId;
    private String reviewContent;
    private String reviewCount;
    private String reviewCreateDate;
    private int reviewId;
    private String reviewImage;
    private int reviewLikeCount;
    private int reviewMainId;
    private int reviewReplyId;
    private int reviewed;
    private String tag;
    private String title;
    private Integer userGender;
    private String userHeadIcon;
    private Integer userId;
    private int userKing;
    private String userName;
    private String viewCount;
    private Integer viewed;

    public String getCollectCount() {
        return this.collectCount;
    }

    public String getCollectDate() {
        return this.collectDate;
    }

    public int getCollectStatus() {
        return this.collectStatus;
    }

    public Integer getCollected() {
        return this.collected;
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

    public int getHasSex() {
        return this.hasSex;
    }

    public Integer getId() {
        return this.id;
    }

    public ArrayList<String> getImages() {
        return this.images;
    }

    public int getIsFans() {
        return this.isFans;
    }

    public String getLikeCount() {
        return this.likeCount;
    }

    public int getLikeStatus() {
        return this.likeStatus;
    }

    public Integer getLikeed() {
        return this.likeed;
    }

    public String getLocalId() {
        return this.localId;
    }

    public String getReviewContent() {
        return this.reviewContent;
    }

    public String getReviewCount() {
        return this.reviewCount;
    }

    public String getReviewCreateDate() {
        return this.reviewCreateDate;
    }

    public int getReviewId() {
        return this.reviewId;
    }

    public String getReviewImage() {
        return this.reviewImage;
    }

    public int getReviewLikeCount() {
        return this.reviewLikeCount;
    }

    public int getReviewMainId() {
        return this.reviewMainId;
    }

    public int getReviewReplyId() {
        return this.reviewReplyId;
    }

    public int getReviewed() {
        return this.reviewed;
    }

    public String getTag() {
        return this.tag;
    }

    public String getTitle() {
        return this.title;
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

    public int getUserKing() {
        return this.userKing;
    }

    public String getUserName() {
        return this.userName;
    }

    public String getViewCount() {
        return this.viewCount;
    }

    public Integer getViewed() {
        return this.viewed;
    }

    public void setCollectCount(String str) {
        this.collectCount = str;
    }

    public void setCollectDate(String str) {
        this.collectDate = str;
    }

    public void setCollectStatus(int i) {
        this.collectStatus = i;
    }

    public void setCollected(Integer num) {
        this.collected = num;
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

    public void setHasSex(int i) {
        this.hasSex = i;
    }

    public void setId(Integer num) {
        this.id = num;
    }

    public void setImages(ArrayList<String> arrayList) {
        this.images = arrayList;
    }

    public void setIsFans(int i) {
        this.isFans = i;
    }

    public void setLikeCount(String str) {
        this.likeCount = str;
    }

    public void setLikeStatus(int i) {
        this.likeStatus = i;
    }

    public void setLikeed(Integer num) {
        this.likeed = num;
    }

    public void setLocalId(String str) {
        this.localId = str;
    }

    public void setReviewContent(String str) {
        this.reviewContent = str;
    }

    public void setReviewCount(String str) {
        this.reviewCount = str;
    }

    public void setReviewCreateDate(String str) {
        this.reviewCreateDate = str;
    }

    public void setReviewId(int i) {
        this.reviewId = i;
    }

    public void setReviewImage(String str) {
        this.reviewImage = str;
    }

    public void setReviewLikeCount(int i) {
        this.reviewLikeCount = i;
    }

    public void setReviewMainId(int i) {
        this.reviewMainId = i;
    }

    public void setReviewReplyId(int i) {
        this.reviewReplyId = i;
    }

    public void setReviewed(int i) {
        this.reviewed = i;
    }

    public void setTag(String str) {
        this.tag = str;
    }

    public void setTitle(String str) {
        this.title = str;
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

    public void setUserKing(int i) {
        this.userKing = i;
    }

    public void setUserName(String str) {
        this.userName = str;
    }

    public void setViewCount(String str) {
        this.viewCount = str;
    }

    public void setViewed(Integer num) {
        this.viewed = num;
    }

    public String toString() {
        return "TopicModel{id=" + this.id + ", localId='" + this.localId + "', title='" + this.title + "', content='" + this.content + "', tag='" + this.tag + "', images=" + this.images + ", createDate='" + this.createDate + "', collectDate='" + this.collectDate + "', likeCount='" + this.likeCount + "', likeed=" + this.likeed + ", reviewCount='" + this.reviewCount + "', reviewed=" + this.reviewed + ", collectCount='" + this.collectCount + "', collected=" + this.collected + ", viewCount='" + this.viewCount + "', viewed=" + this.viewed + ", userId=" + this.userId + ", userKing=" + this.userKing + ", gradeIcon='" + this.gradeIcon + "', userName='" + this.userName + "', userGender=" + this.userGender + ", userHeadIcon='" + this.userHeadIcon + "', isFans=" + this.isFans + ", reviewId=" + this.reviewId + ", reviewMainId=" + this.reviewMainId + ", reviewReplyId=" + this.reviewReplyId + ", reviewImage='" + this.reviewImage + "', reviewContent='" + this.reviewContent + "', reviewCreateDate='" + this.reviewCreateDate + "', reviewLikeCount=" + this.reviewLikeCount + ", likeStatus=" + this.likeStatus + ", collectStatus=" + this.collectStatus + ", hasSex=" + this.hasSex + '}';
    }
}
