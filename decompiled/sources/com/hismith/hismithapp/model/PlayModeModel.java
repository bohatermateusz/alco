package com.hismith.hismithapp.model;

import android.os.Parcel;
import android.os.Parcelable;
import android.text.TextUtils;
import com.google.gson.Gson;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class PlayModeModel implements Parcelable {
    public static final Parcelable.Creator<PlayModeModel> CREATOR = new Parcelable.Creator<PlayModeModel>() { // from class: com.hismith.hismithapp.model.PlayModeModel.1
        /* JADX WARN: Can't rename method to resolve collision */
        @Override // android.os.Parcelable.Creator
        public PlayModeModel createFromParcel(Parcel parcel) {
            PlayModeModel playModeModel = new PlayModeModel();
            playModeModel.setModeId(parcel.readInt());
            playModeModel.setModeName(parcel.readString());
            playModeModel.setCode(parcel.readString());
            playModeModel.setProductLogo(parcel.readString());
            playModeModel.setCreateDate(parcel.readString());
            playModeModel.setPlayCount(parcel.readString());
            playModeModel.setLikeCount(parcel.readString());
            playModeModel.setLiked(parcel.readInt());
            playModeModel.setCollectCount(parcel.readString());
            playModeModel.setCollected(parcel.readInt());
            playModeModel.setShared(parcel.readInt());
            playModeModel.setModeData(parcel.readString());
            playModeModel.setModePoints(parcel.readString());
            playModeModel.setModeSetting(parcel.readString());
            playModeModel.setModeTime(parcel.readString());
            playModeModel.setUserId(parcel.readInt());
            playModeModel.setUserKing(parcel.readInt());
            playModeModel.setGradeIcon(parcel.readString());
            playModeModel.setUserName(parcel.readString());
            playModeModel.setUserGender(parcel.readInt());
            playModeModel.setUserHeadIcon(parcel.readString());
            playModeModel.setIsFollow(parcel.readInt());
            playModeModel.setPlayTime(parcel.readFloat());
            return playModeModel;
        }

        /* JADX WARN: Can't rename method to resolve collision */
        @Override // android.os.Parcelable.Creator
        public PlayModeModel[] newArray(int i) {
            return new PlayModeModel[i];
        }
    };
    private String code;
    private String collectCount;
    private int collected;
    private String createDate;
    private String gradeIcon;
    private int isFollow;
    private String likeCount;
    private int liked;
    private String modeData;
    private int modeId;
    private String modeName;
    private String modePoints;
    private String modeSetting;
    private ModeSettingModel modeSettingsModel;
    private String modeTime;
    private String playCount;
    private float playTime;
    private String productLogo;
    private int shared;
    private int userGender;
    private String userHeadIcon;
    private int userId;
    private int userKing;
    private String userName;

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    public String getCode() {
        return this.code;
    }

    public String getCollectCount() {
        return this.collectCount;
    }

    public int getCollected() {
        return this.collected;
    }

    public String getCreateDate() {
        return this.createDate;
    }

    public String getGradeIcon() {
        return this.gradeIcon;
    }

    public int getIsFollow() {
        return this.isFollow;
    }

    public String getLikeCount() {
        return this.likeCount;
    }

    public int getLiked() {
        return this.liked;
    }

    public String getModeData() {
        return this.modeData;
    }

    public int getModeId() {
        return this.modeId;
    }

    public String getModeName() {
        return this.modeName;
    }

    public String getModePoints() {
        return this.modePoints;
    }

    public String getModeSetting() {
        return this.modeSetting;
    }

    public ModeSettingModel getModeSettingsModel() {
        return this.modeSettingsModel;
    }

    public String getModeTime() {
        return this.modeTime;
    }

    public String getPlayCount() {
        return this.playCount;
    }

    public float getPlayTime() {
        return this.playTime;
    }

    public String getProductLogo() {
        return this.productLogo;
    }

    public int getShared() {
        return this.shared;
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

    public void initModeSettingsModel() {
        LogUtil.e("重载模式 modeSetting:" + this.modeSetting);
        if (TextUtils.isEmpty(this.modeSetting)) {
            return;
        }
        this.modeSettingsModel = (ModeSettingModel) new Gson().fromJson(this.modeSetting, ModeSettingModel.class);
    }

    public void setCode(String str) {
        this.code = str;
    }

    public void setCollectCount(String str) {
        this.collectCount = str;
    }

    public void setCollected(int i) {
        this.collected = i;
    }

    public void setCreateDate(String str) {
        this.createDate = str;
    }

    public void setGradeIcon(String str) {
        this.gradeIcon = str;
    }

    public void setIsFollow(int i) {
        this.isFollow = i;
    }

    public void setLikeCount(String str) {
        this.likeCount = str;
    }

    public void setLiked(int i) {
        this.liked = i;
    }

    public void setModeData(String str) {
        this.modeData = str;
    }

    public void setModeId(int i) {
        this.modeId = i;
    }

    public void setModeName(String str) {
        this.modeName = str;
    }

    public void setModePoints(String str) {
        this.modePoints = str;
    }

    public void setModeSetting(String str) {
        this.modeSetting = str;
    }

    public void setModeTime(String str) {
        this.modeTime = str;
    }

    public void setPlayCount(String str) {
        this.playCount = str;
    }

    public void setPlayTime(float f) {
        this.playTime = f;
    }

    public void setProductLogo(String str) {
        this.productLogo = str;
    }

    public void setShared(int i) {
        this.shared = i;
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
        return "PlayModeModel{modeId=" + this.modeId + ", modeName='" + this.modeName + "', code='" + this.code + "', productLogo='" + this.productLogo + "', createDate='" + this.createDate + "', playCount=" + this.playCount + ", likeCount=" + this.likeCount + ", liked=" + this.liked + ", collectCount=" + this.collectCount + ", collected=" + this.collected + ", shared=" + this.shared + ", modeData='" + this.modeData + "', modePoints='" + this.modePoints + "', modeSetting='" + this.modeSetting + "', modeTime='" + this.modeTime + "', userId=" + this.userId + ", userKing=" + this.userKing + ", gradeIcon='" + this.gradeIcon + "', userName='" + this.userName + "', userGender=" + this.userGender + ", userHeadIcon='" + this.userHeadIcon + "', isFollow=" + this.isFollow + ", playTime=" + this.playTime + '}';
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeInt(this.modeId);
        parcel.writeString(this.modeName);
        parcel.writeString(this.code);
        parcel.writeString(this.productLogo);
        parcel.writeString(this.createDate);
        parcel.writeString(this.playCount);
        parcel.writeString(this.likeCount);
        parcel.writeInt(this.liked);
        parcel.writeString(this.collectCount);
        parcel.writeInt(this.collected);
        parcel.writeInt(this.shared);
        parcel.writeString(this.modeData);
        parcel.writeString(this.modePoints);
        parcel.writeString(this.modeSetting);
        parcel.writeString(this.modeTime);
        parcel.writeInt(this.userId);
        parcel.writeInt(this.userKing);
        parcel.writeString(this.gradeIcon);
        parcel.writeString(this.userName);
        parcel.writeInt(this.userGender);
        parcel.writeString(this.userHeadIcon);
        parcel.writeInt(this.isFollow);
        parcel.writeFloat(this.playTime);
    }
}
