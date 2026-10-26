package com.hismith.hismithapp.model.lomi;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class SocketResponseModel extends HttpModel {
    private String action;
    private SocketResultModel result;

    public String getAction() {
        return this.action;
    }

    public SocketResultModel getResult() {
        return this.result;
    }

    public void setAction(String str) {
        this.action = str;
    }

    public void setResult(SocketResultModel socketResultModel) {
        this.result = socketResultModel;
    }
}
