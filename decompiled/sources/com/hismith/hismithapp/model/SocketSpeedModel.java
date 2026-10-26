package com.hismith.hismithapp.model;

import com.google.gson.JsonElement;
import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class SocketSpeedModel extends HttpModel {
    private String action;
    private int clientConnection;
    private int climax;
    private int funId;
    private String gameToken;
    private JsonElement lines;
    private int mode;
    private int offline;
    private int onlyRead;
    private SocketProductModel product;
    private String remoteUrl;
    private int reversal;
    private int serverConnection;
    private int smooth;
    private int speed;
    private int status;
    private Integer strokeEnd;
    private int strokeMax;
    private int strokeMin;
    private int strokeNow;
    private Integer strokeStart;
    private int strokeType;
    private String type;

    public class Modes {
        private String[] modes;
        private String[] vibrations;

        public Modes() {
        }

        public String[] getModes() {
            return this.modes;
        }

        public String[] getVibrations() {
            return this.vibrations;
        }

        public void setModes(String[] strArr) {
            this.modes = strArr;
        }

        public void setVibrations(String[] strArr) {
            this.vibrations = strArr;
        }
    }

    public String getAction() {
        return this.action;
    }

    public int getClientConnection() {
        return this.clientConnection;
    }

    public int getClimax() {
        return this.climax;
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

    public int getMode() {
        return this.mode;
    }

    public int getOffline() {
        return this.offline;
    }

    public int getOnlyRead() {
        return this.onlyRead;
    }

    public SocketProductModel getProduct() {
        return this.product;
    }

    public String getRemoteUrl() {
        return this.remoteUrl;
    }

    public int getReversal() {
        return this.reversal;
    }

    public int getServerConnection() {
        return this.serverConnection;
    }

    public int getSmooth() {
        return this.smooth;
    }

    public int getSpeed() {
        return this.speed;
    }

    public int getStatus() {
        return this.status;
    }

    public Integer getStrokeEnd() {
        return this.strokeEnd;
    }

    public int getStrokeMax() {
        return this.strokeMax;
    }

    public int getStrokeMin() {
        return this.strokeMin;
    }

    public int getStrokeNow() {
        return this.strokeNow;
    }

    public Integer getStrokeStart() {
        return this.strokeStart;
    }

    public int getStrokeType() {
        return this.strokeType;
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

    public void setClientConnection(int i) {
        this.clientConnection = i;
    }

    public void setClimax(int i) {
        this.climax = i;
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

    public void setMode(int i) {
        this.mode = i;
    }

    public void setOffline(int i) {
        this.offline = i;
    }

    public void setOnlyRead(int i) {
        this.onlyRead = i;
    }

    public void setProduct(SocketProductModel socketProductModel) {
        this.product = socketProductModel;
    }

    public void setRemoteUrl(String str) {
        this.remoteUrl = str;
    }

    public void setReversal(int i) {
        this.reversal = i;
    }

    public void setServerConnection(int i) {
        this.serverConnection = i;
    }

    public void setSmooth(int i) {
        this.smooth = i;
    }

    public void setSpeed(int i) {
        this.speed = i;
    }

    public void setStatus(int i) {
        this.status = i;
    }

    public void setStrokeEnd(Integer num) {
        this.strokeEnd = num;
    }

    public void setStrokeMax(int i) {
        this.strokeMax = i;
    }

    public void setStrokeMin(int i) {
        this.strokeMin = i;
    }

    public void setStrokeNow(int i) {
        this.strokeNow = i;
    }

    public void setStrokeStart(Integer num) {
        this.strokeStart = num;
    }

    public void setStrokeType(int i) {
        this.strokeType = i;
    }

    public void setType(String str) {
        this.type = str;
    }
}
