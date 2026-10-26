package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class FirmwareModel extends HttpModel {
    private String action;
    private String fwData;
    private String fwPass;
    private String fwVersion;
    private String productId;

    public String getAction() {
        return this.action;
    }

    public String getFwData() {
        return this.fwData;
    }

    public String getFwPass() {
        return this.fwPass;
    }

    public String getFwVersion() {
        return this.fwVersion;
    }

    public String getProductId() {
        return this.productId;
    }

    public void setAction(String str) {
        this.action = str;
    }

    public void setFwData(String str) {
        this.fwData = str;
    }

    public void setFwPass(String str) {
        this.fwPass = str;
    }

    public void setFwVersion(String str) {
        this.fwVersion = str;
    }

    public void setProductId(String str) {
        this.productId = str;
    }

    public String toString() {
        return "FirmwareModel{action='" + this.action + "', fwVersion='" + this.fwVersion + "', fwData='" + this.fwData + "', fwPass='" + this.fwPass + "', productId='" + this.productId + "'}";
    }
}
