package com.hismith.hismithapp.model;

import android.os.Parcel;
import android.os.Parcelable;

/* loaded from: classes3.dex */
public class FirstChatModel implements Parcelable {
    public static final Parcelable.Creator<FirstChatModel> CREATOR = new Parcelable.Creator<FirstChatModel>() { // from class: com.hismith.hismithapp.model.FirstChatModel.1
        /* JADX WARN: Can't rename method to resolve collision */
        @Override // android.os.Parcelable.Creator
        public FirstChatModel createFromParcel(Parcel parcel) {
            FirstChatModel firstChatModel = new FirstChatModel();
            firstChatModel.setUserId(parcel.readInt());
            firstChatModel.setKing(parcel.readInt());
            firstChatModel.setGradeIcon(parcel.readString());
            firstChatModel.setName(parcel.readString());
            firstChatModel.setGender(parcel.readInt());
            firstChatModel.setAge(parcel.readString());
            firstChatModel.setConstellation(parcel.readString());
            firstChatModel.setHeadIcon(parcel.readString());
            firstChatModel.setType(parcel.readString());
            firstChatModel.setContent(parcel.readString());
            firstChatModel.setVoiceLength(parcel.readInt());
            firstChatModel.setDate(parcel.readString());
            return firstChatModel;
        }

        /* JADX WARN: Can't rename method to resolve collision */
        @Override // android.os.Parcelable.Creator
        public FirstChatModel[] newArray(int i) {
            return new FirstChatModel[i];
        }
    };
    private String age;
    private String constellation;
    private String content;
    private String date;
    private int gender;
    private String gradeIcon;
    private String headIcon;
    private int king;
    private String name;
    private String type;
    private int userId;
    private int voiceLength;

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    public String getAge() {
        return this.age;
    }

    public String getConstellation() {
        return this.constellation;
    }

    public String getContent() {
        return this.content;
    }

    public String getDate() {
        return this.date;
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

    public int getKing() {
        return this.king;
    }

    public String getName() {
        return this.name;
    }

    public String getType() {
        return this.type;
    }

    public int getUserId() {
        return this.userId;
    }

    public int getVoiceLength() {
        return this.voiceLength;
    }

    public void setAge(String str) {
        this.age = str;
    }

    public void setConstellation(String str) {
        this.constellation = str;
    }

    public void setContent(String str) {
        this.content = str;
    }

    public void setDate(String str) {
        this.date = str;
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

    public void setKing(int i) {
        this.king = i;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setType(String str) {
        this.type = str;
    }

    public void setUserId(int i) {
        this.userId = i;
    }

    public void setVoiceLength(int i) {
        this.voiceLength = i;
    }

    public String toString() {
        return "FirstChat{userId=" + this.userId + ", king=" + this.king + ", gradeIcon='" + this.gradeIcon + "', name='" + this.name + "', gender=" + this.gender + ", age='" + this.age + "', constellation='" + this.constellation + "', headIcon='" + this.headIcon + "', type='" + this.type + "', content='" + this.content + "', voiceLength=" + this.voiceLength + ", date='" + this.date + "'}";
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeInt(this.userId);
        parcel.writeInt(this.king);
        parcel.writeString(this.gradeIcon);
        parcel.writeString(this.name);
        parcel.writeInt(this.gender);
        parcel.writeString(this.age);
        parcel.writeString(this.constellation);
        parcel.writeString(this.headIcon);
        parcel.writeString(this.type);
        parcel.writeString(this.content);
        parcel.writeInt(this.voiceLength);
        parcel.writeString(this.date);
    }
}
