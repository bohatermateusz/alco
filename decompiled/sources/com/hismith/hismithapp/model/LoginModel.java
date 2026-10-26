package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class LoginModel extends HttpModel {
    private String name;
    private String token;
    private String userLangIso;

    public String getName() {
        return this.name;
    }

    public String getToken() {
        return this.token;
    }

    public String getUserLangIso() {
        return this.userLangIso;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setToken(String str) {
        this.token = str;
    }

    public void setUserLangIso(String str) {
        this.userLangIso = str;
    }

    public String toString() {
        return "LoginModel{token='" + this.token + "', name='" + this.name + "', userLangIso='" + this.userLangIso + "'}";
    }
}
