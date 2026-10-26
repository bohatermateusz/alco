package com.hismith.hismithapp.model;

import com.google.gson.JsonElement;

/* loaded from: classes3.dex */
public class RemotePlayModel {
    private String action;
    private String clientConnection;
    private String errorCode;
    private String errorMessage;
    private int funId;
    private String gameToken;
    private JsonElement lines;
    private String message;
    private int mode;
    private String offline;
    private int onlyRead;
    private String remoteUrl;
    private int reversal;
    private int speed;
    private Integer strokeEnd;
    private int strokeNow;
    private Integer strokeStart;
    private int success;
    private String type;

    class Line {
        private int x;
        private int y;

        Line() {
        }

        public int getX() {
            return this.x;
        }

        public int getY() {
            return this.y;
        }

        public void setX(int i) {
            this.x = i;
        }

        public void setY(int i) {
            this.y = i;
        }

        public String toString() {
            return "Line{x=" + this.x + ", y=" + this.y + '}';
        }
    }

    public String getAction() {
        return this.action;
    }

    public String getClientConnection() {
        return this.clientConnection;
    }

    public String getErrorCode() {
        return this.errorCode;
    }

    public String getErrorMessage() {
        return this.errorMessage;
    }

    public int getFunId() {
        return this.funId;
    }

    public String getGameToken() {
        return this.gameToken;
    }

    public JsonElement getLines() {
        return this.lines;
    }

    public String getMessage() {
        return this.message;
    }

    public int getMode() {
        return this.mode;
    }

    public String getOffline() {
        return this.offline;
    }

    public int getOnlyRead() {
        return this.onlyRead;
    }

    public String getRemoteUrl() {
        return this.remoteUrl;
    }

    public int getReversal() {
        return this.reversal;
    }

    public int getSpeed() {
        return this.speed;
    }

    public Integer getStrokeEnd() {
        return this.strokeEnd;
    }

    public int getStrokeNow() {
        return this.strokeNow;
    }

    public Integer getStrokeStart() {
        return this.strokeStart;
    }

    public int getSuccess() {
        return this.success;
    }

    public String getType() {
        return this.type;
    }

    public boolean isMotor1() {
        return 1 == this.funId;
    }

    public boolean isMotor1Mode() {
        return 1 == this.funId;
    }

    public boolean isMotor2() {
        return 2 == this.funId;
    }

    public boolean isMotor2Mode() {
        return 2 == this.funId;
    }

    public void setAction(String str) {
        this.action = str;
    }

    public void setClientConnection(String str) {
        this.clientConnection = str;
    }

    public void setErrorCode(String str) {
        this.errorCode = str;
    }

    public void setErrorMessage(String str) {
        this.errorMessage = str;
    }

    public void setFunId(int i) {
        this.funId = i;
    }

    public void setGameToken(String str) {
        this.gameToken = str;
    }

    public void setLines(JsonElement jsonElement) {
        this.lines = jsonElement;
    }

    public void setMessage(String str) {
        this.message = str;
    }

    public void setMode(int i) {
        this.mode = i;
    }

    public void setOffline(String str) {
        this.offline = str;
    }

    public void setOnlyRead(int i) {
        this.onlyRead = i;
    }

    public void setRemoteUrl(String str) {
        this.remoteUrl = str;
    }

    public void setReversal(int i) {
        this.reversal = i;
    }

    public void setSpeed(int i) {
        this.speed = i;
    }

    public void setStrokeEnd(Integer num) {
        this.strokeEnd = num;
    }

    public void setStrokeNow(int i) {
        this.strokeNow = i;
    }

    public void setStrokeStart(Integer num) {
        this.strokeStart = num;
    }

    public void setSuccess(int i) {
        this.success = i;
    }

    public void setType(String str) {
        this.type = str;
    }

    public String toString() {
        return "RemotePlayModel{gameToken='" + this.gameToken + "', remoteUrl='" + this.remoteUrl + "', action='" + this.action + "', speed=" + this.speed + ", mode=" + this.mode + ", type='" + this.type + "', funId=" + this.funId + ", strokeNow=" + this.strokeNow + ", reversal=" + this.reversal + ", message='" + this.message + "', strokeStart=" + this.strokeStart + ", strokeEnd=" + this.strokeEnd + ", onlyRead=" + this.onlyRead + ", lines=" + this.lines + ", clientConnection='" + this.clientConnection + "', offline='" + this.offline + "', success=" + this.success + ", errorCode='" + this.errorCode + "', errorMessage='" + this.errorMessage + "'}";
    }
}
