package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class LanguageModel {
    private String iso;
    private String name;
    private String version;

    public LanguageModel() {
    }

    public LanguageModel(String str, String str2) {
        this.iso = str;
        this.name = str2;
    }

    public LanguageModel(String str, String str2, String str3) {
        this.iso = str;
        this.version = str2;
        this.name = str3;
    }

    public String getIso() {
        return this.iso;
    }

    public String getName() {
        return this.name;
    }

    public String getVersion() {
        return this.version;
    }

    public void setIso(String str) {
        this.iso = str;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setVersion(String str) {
        this.version = str;
    }
}
