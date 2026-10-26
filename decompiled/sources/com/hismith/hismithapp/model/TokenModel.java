package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class TokenModel {
    private String access_token;
    private int expires_in;
    private String id_token;
    private String scope;
    private String token_type;

    public String getAccess_token() {
        return this.access_token;
    }

    public int getExpires_in() {
        return this.expires_in;
    }

    public String getId_token() {
        return this.id_token;
    }

    public String getScope() {
        return this.scope;
    }

    public String getToken_type() {
        return this.token_type;
    }

    public void setAccess_token(String str) {
        this.access_token = str;
    }

    public void setExpires_in(int i) {
        this.expires_in = i;
    }

    public void setId_token(String str) {
        this.id_token = str;
    }

    public void setScope(String str) {
        this.scope = str;
    }

    public void setToken_type(String str) {
        this.token_type = str;
    }

    public String toString() {
        return "TokenModel{access_token='" + this.access_token + "', expires_in=" + this.expires_in + ", scope='" + this.scope + "', token_type='" + this.token_type + "', id_token='" + this.id_token + "'}";
    }
}
