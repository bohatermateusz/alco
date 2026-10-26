package com.hismith.hismithapp.model;

import com.google.gson.JsonElement;
import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class CheckUpdateModel extends HttpModel {
    private String aboutApp;
    private JsonElement allLangs;
    private JsonElement complaintReasons;
    private int contactUserId;
    private String copyright;
    private String mainBackgroundUrl;
    private String newVersion;
    private String newVersionDesc;
    private int newVersionForcibly;
    private int newVersionPrompt;
    private String roomVoice;
    private Integer storeStatus;
    private String storeUrl;
    private String token;
    private String userLangIso;
    private String userTermsVersion;

    public String getAboutApp() {
        return this.aboutApp;
    }

    public JsonElement getAllLangs() {
        return this.allLangs;
    }

    public JsonElement getComplaintReasons() {
        return this.complaintReasons;
    }

    public int getContactUserId() {
        return this.contactUserId;
    }

    public String getCopyright() {
        return this.copyright;
    }

    public String getMainBackgroundUrl() {
        return this.mainBackgroundUrl;
    }

    public String getNewVersion() {
        return this.newVersion;
    }

    public String getNewVersionDesc() {
        return this.newVersionDesc;
    }

    public int getNewVersionForcibly() {
        return this.newVersionForcibly;
    }

    public int getNewVersionPrompt() {
        return this.newVersionPrompt;
    }

    public String getRoomVoice() {
        return this.roomVoice;
    }

    public Integer getStoreStatus() {
        return this.storeStatus;
    }

    public String getStoreUrl() {
        return this.storeUrl;
    }

    public String getToken() {
        return this.token;
    }

    public String getUserLangIso() {
        return this.userLangIso;
    }

    public String getUserTermsVersion() {
        return this.userTermsVersion;
    }

    public void setAboutApp(String str) {
        this.aboutApp = str;
    }

    public void setAllLangs(JsonElement jsonElement) {
        this.allLangs = jsonElement;
    }

    public void setComplaintReasons(JsonElement jsonElement) {
        this.complaintReasons = jsonElement;
    }

    public void setContactUserId(int i) {
        this.contactUserId = i;
    }

    public void setCopyright(String str) {
        this.copyright = str;
    }

    public void setMainBackgroundUrl(String str) {
        this.mainBackgroundUrl = str;
    }

    public void setNewVersion(String str) {
        this.newVersion = str;
    }

    public void setNewVersionDesc(String str) {
        this.newVersionDesc = str;
    }

    public void setNewVersionForcibly(int i) {
        this.newVersionForcibly = i;
    }

    public void setNewVersionPrompt(int i) {
        this.newVersionPrompt = i;
    }

    public void setRoomVoice(String str) {
        this.roomVoice = str;
    }

    public void setStoreStatus(Integer num) {
        this.storeStatus = num;
    }

    public void setStoreUrl(String str) {
        this.storeUrl = str;
    }

    public void setToken(String str) {
        this.token = str;
    }

    public void setUserLangIso(String str) {
        this.userLangIso = str;
    }

    public void setUserTermsVersion(String str) {
        this.userTermsVersion = str;
    }

    public String toString() {
        return "CheckUpdateModel{storeStatus=" + this.storeStatus + ", storeUrl='" + this.storeUrl + "', roomVoice='" + this.roomVoice + "', contactUserId=" + this.contactUserId + ", allLangs=" + this.allLangs + ", userTermsVersion='" + this.userTermsVersion + "', mainBackgroundUrl='" + this.mainBackgroundUrl + "', token='" + this.token + "', newVersion='" + this.newVersion + "', newVersionForcibly=" + this.newVersionForcibly + ", newVersionPrompt=" + this.newVersionPrompt + ", newVersionDesc='" + this.newVersionDesc + "', aboutApp='" + this.aboutApp + "', copyright='" + this.copyright + "', complaintReasons=" + this.complaintReasons + ", userLangIso='" + this.userLangIso + "'}";
    }
}
