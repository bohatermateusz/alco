package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class TopicDraftModel {
    private String categorieId;
    private String content;
    private String[] contentPhoto;
    private String contentText;
    private String date;
    private int id;
    private String title;
    private int userId;

    public String getCategorieId() {
        return this.categorieId;
    }

    public String getContent() {
        return this.content;
    }

    public String[] getContentPhoto() {
        return this.contentPhoto;
    }

    public String getContentText() {
        return this.contentText;
    }

    public String getDate() {
        return this.date;
    }

    public int getId() {
        return this.id;
    }

    public String getTitle() {
        return this.title;
    }

    public int getUserId() {
        return this.userId;
    }

    public void setCategorieId(String str) {
        this.categorieId = str;
    }

    public void setContent(String str) {
        this.content = str;
    }

    public void setContentPhoto(String[] strArr) {
        this.contentPhoto = strArr;
    }

    public void setContentText(String str) {
        this.contentText = str;
    }

    public void setDate(String str) {
        this.date = str;
    }

    public void setId(int i) {
        this.id = i;
    }

    public void setTitle(String str) {
        this.title = str;
    }

    public void setUserId(int i) {
        this.userId = i;
    }
}
