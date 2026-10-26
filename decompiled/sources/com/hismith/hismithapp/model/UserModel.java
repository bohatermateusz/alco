package com.hismith.hismithapp.model;

import android.text.TextUtils;
import com.hismith.hismithapp.http.HttpModel;
import java.io.Serializable;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class UserModel extends HttpModel implements Serializable, Cloneable {
    private int addFriendRemind;
    private String adsContent;
    private String adsImage;
    private String adsVersion;
    private String alias;
    private String bios;
    private String birthday;
    private int chatUpState;
    private String city;
    private int cityId;
    private String constellation;
    private String country;
    private int countryId;
    private String email;
    private int fansCount;
    private int focusCount;
    private int focusRemind;
    private int friendCount;
    private int gender;
    private String gradeIcon;
    private int hasChat;
    private String headIcon;
    private int headSex;
    private int interestedIn = -1;
    private int isBlocked;
    private int isFans;
    private int isFriend;
    private int king;
    private String level;
    private int messageRemind;
    private int mutuallyRemind;
    private String name;
    private String phone;
    private String photos;
    private int playLogs;
    private int productAds;
    private String sourceId;
    private int stealth;
    private int topicCount;
    private UserPhotosModel[] uPhotos;
    private int userId;
    private int viewSex;
    private String voice;
    private int voiceLength;

    /* renamed from: clone, reason: merged with bridge method [inline-methods] */
    public UserModel m614clone() {
        try {
            return (UserModel) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
            return null;
        }
    }

    public int getAddFriendRemind() {
        return this.addFriendRemind;
    }

    public String getAdsContent() {
        return this.adsContent;
    }

    public String getAdsImage() {
        return this.adsImage;
    }

    public String getAdsVersion() {
        return this.adsVersion;
    }

    public String getAlias() {
        return !TextUtils.isEmpty(this.alias) ? this.alias : this.name;
    }

    public String getBios() {
        return this.bios;
    }

    public String getBirthday() {
        return this.birthday;
    }

    public int getChatUpState() {
        return this.chatUpState;
    }

    public String getCity() {
        return this.city;
    }

    public int getCityId() {
        return this.cityId;
    }

    public String getConstellation() {
        return this.constellation;
    }

    public String getCountry() {
        return this.country;
    }

    public int getCountryId() {
        return this.countryId;
    }

    public String getEmail() {
        return this.email;
    }

    public int getFansCount() {
        return this.fansCount;
    }

    public int getFocusCount() {
        return this.focusCount;
    }

    public int getFocusRemind() {
        return this.focusRemind;
    }

    public int getFriendCount() {
        return this.friendCount;
    }

    public int getGender() {
        return this.gender;
    }

    public String getGradeIcon() {
        return this.gradeIcon;
    }

    public int getHasChat() {
        return this.hasChat;
    }

    public String getHeadIcon() {
        return this.headIcon;
    }

    public int getHeadSex() {
        return this.headSex;
    }

    public int getInterestedIn() {
        return this.interestedIn;
    }

    public int getIsBlocked() {
        return this.isBlocked;
    }

    public int getIsFans() {
        return this.isFans;
    }

    public int getIsFriend() {
        return this.isFriend;
    }

    public int getKing() {
        return this.king;
    }

    public String getLevel() {
        return this.level;
    }

    public int getMessageRemind() {
        return this.messageRemind;
    }

    public int getMutuallyRemind() {
        return this.mutuallyRemind;
    }

    public String getName() {
        return this.name;
    }

    public String getPhone() {
        return this.phone;
    }

    public String getPhotos() {
        return this.photos;
    }

    public int getPlayLogs() {
        return this.playLogs;
    }

    public int getProductAds() {
        return this.productAds;
    }

    public String getSourceId() {
        return this.sourceId;
    }

    public int getStealth() {
        return this.stealth;
    }

    public int getTopicCount() {
        return this.topicCount;
    }

    public int getUserId() {
        return this.userId;
    }

    public int getViewSex() {
        return this.viewSex;
    }

    public String getVoice() {
        return this.voice;
    }

    public int getVoiceLength() {
        return this.voiceLength;
    }

    public UserPhotosModel[] getuPhotos() {
        return this.uPhotos;
    }

    public void setAddFriendRemind(int i) {
        this.addFriendRemind = i;
    }

    public void setAdsContent(String str) {
        this.adsContent = str;
    }

    public void setAdsImage(String str) {
        this.adsImage = str;
    }

    public void setAdsVersion(String str) {
        this.adsVersion = str;
    }

    public void setAlias(String str) {
        this.alias = str;
    }

    public void setBios(String str) {
        this.bios = str;
    }

    public void setBirthday(String str) {
        this.birthday = str;
    }

    public void setChatUpState(int i) {
        this.chatUpState = i;
    }

    public void setCity(String str) {
        this.city = str;
    }

    public void setCityId(int i) {
        this.cityId = i;
    }

    public void setConstellation(String str) {
        this.constellation = str;
    }

    public void setCountry(String str) {
        this.country = str;
    }

    public void setCountryId(int i) {
        this.countryId = i;
    }

    public void setEmail(String str) {
        this.email = str;
    }

    public void setFansCount(int i) {
        this.fansCount = i;
        if (i < 0) {
            this.fansCount = 0;
        }
    }

    public void setFocusCount(int i) {
        this.focusCount = i;
        if (i < 0) {
            this.focusCount = 0;
        }
    }

    public void setFocusRemind(int i) {
        this.focusRemind = i;
    }

    public void setFriendCount(int i) {
        this.friendCount = i;
        if (i < 0) {
            this.friendCount = 0;
        }
    }

    public void setGender(int i) {
        this.gender = i;
    }

    public void setGradeIcon(String str) {
        this.gradeIcon = str;
    }

    public void setHasChat(int i) {
        this.hasChat = i;
    }

    public void setHeadIcon(String str) {
        this.headIcon = str;
    }

    public void setHeadSex(int i) {
        this.headSex = i;
    }

    public void setInterestedIn(int i) {
        this.interestedIn = i;
    }

    public void setIsBlocked(int i) {
        this.isBlocked = i;
    }

    public void setIsFans(int i) {
        this.isFans = i;
    }

    public void setIsFriend(int i) {
        this.isFriend = i;
    }

    public void setKing(int i) {
        this.king = i;
    }

    public void setLevel(String str) {
        this.level = str;
    }

    public void setMessageRemind(int i) {
        this.messageRemind = i;
    }

    public void setMutuallyRemind(int i) {
        this.mutuallyRemind = i;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setPhone(String str) {
        this.phone = str;
    }

    public void setPhotos(String str) {
        this.photos = str;
    }

    public void setPlayLogs(int i) {
        this.playLogs = i;
    }

    public void setProductAds(int i) {
        this.productAds = i;
    }

    public void setSourceId(String str) {
        this.sourceId = str;
    }

    public void setStealth(int i) {
        this.stealth = i;
    }

    public void setTopicCount(int i) {
        this.topicCount = i;
    }

    public void setUserId(int i) {
        this.userId = i;
    }

    public void setViewSex(int i) {
        this.viewSex = i;
    }

    public void setVoice(String str) {
        this.voice = str;
    }

    public void setVoiceLength(int i) {
        this.voiceLength = i;
    }

    public void setuPhotos(UserPhotosModel[] userPhotosModelArr) {
        this.uPhotos = userPhotosModelArr;
    }

    public String toString() {
        return "UserModel{userId=" + this.userId + ", king=" + this.king + ", gradeIcon='" + this.gradeIcon + "', sourceId='" + this.sourceId + "', name='" + this.name + "', gender=" + this.gender + ", interestedIn=" + this.interestedIn + ", countryId=" + this.countryId + ", country='" + this.country + "', cityId=" + this.cityId + ", city='" + this.city + "', fansCount=" + this.fansCount + ", topicCount=" + this.topicCount + ", birthday='" + this.birthday + "', constellation='" + this.constellation + "', headIcon='" + this.headIcon + "', photos='" + this.photos + "', uPhotos=" + Arrays.toString(this.uPhotos) + ", bios='" + this.bios + "', voice='" + this.voice + "', voiceLength=" + this.voiceLength + ", chatUpState=" + this.chatUpState + ", isFans=" + this.isFans + ", isFriend=" + this.isFriend + ", isBlocked=" + this.isBlocked + ", hasChat=" + this.hasChat + ", level='" + this.level + "', email='" + this.email + "', phone='" + this.phone + "', friendCount=" + this.friendCount + ", focusCount=" + this.focusCount + ", stealth=" + this.stealth + ", playLogs=" + this.playLogs + ", focusRemind=" + this.focusRemind + ", messageRemind=" + this.messageRemind + ", mutuallyRemind=" + this.mutuallyRemind + ", addFriendRemind=" + this.addFriendRemind + ", productAds=" + this.productAds + ", adsVersion='" + this.adsVersion + "', adsImage='" + this.adsImage + "', adsContent='" + this.adsContent + "', viewSex=" + this.viewSex + ", headSex=" + this.headSex + ", alias='" + this.alias + "'}";
    }
}
