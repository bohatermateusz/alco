package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class PlayLogsModel extends HttpModel {
    private String overMsg;
    private String playDate;
    private String playLong;
    private String productId;
    private String productLogo;

    public String getOverMsg() {
        return this.overMsg;
    }

    public String getPlayDate() {
        return this.playDate;
    }

    public String getPlayLong() {
        return this.playLong;
    }

    public String getProductId() {
        return this.productId;
    }

    public String getProductLogo() {
        return this.productLogo;
    }

    public void setOverMsg(String str) {
        this.overMsg = str;
    }

    public void setPlayDate(String str) {
        this.playDate = str;
    }

    public void setPlayLong(String str) {
        this.playLong = str;
    }

    public void setProductId(String str) {
        this.productId = str;
    }

    public void setProductLogo(String str) {
        this.productLogo = str;
    }

    public String toString() {
        return "PlayLogsModel{overMsg='" + this.overMsg + "', playLong='" + this.playLong + "', playDate='" + this.playDate + "', productId='" + this.productId + "', productLogo='" + this.productLogo + "'}";
    }
}
