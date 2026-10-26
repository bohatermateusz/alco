package com.hismith.hismithapp.model;

import java.io.Serializable;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class SocketProductModel implements Serializable {
    private String climax;
    private int mode;
    private int modeStatus;
    private Modes modes;
    private Modes modesUrl;
    private int music;
    private int reversal;
    private int reversalFun;
    private int shake;
    private int slide;
    private int smooth;
    private int speed;
    private int strokeMax;
    private int strokeMin;
    private int strokeNow;
    private int strokeType;
    private int svgLeftRight;
    private String svgName;
    private String svgUrl;
    private long svgVersion;
    private String type;
    private int vibration;
    private int vspeed;

    public class Modes implements Serializable {
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

        public String toString() {
            return "Modes{modes=" + Arrays.toString(this.modes) + ", vibrations=" + Arrays.toString(this.vibrations) + '}';
        }
    }

    public String getClimax() {
        return this.climax;
    }

    public int getMode() {
        return this.mode;
    }

    public int getModeStatus() {
        return this.modeStatus;
    }

    public Modes getModes() {
        return this.modes;
    }

    public Modes getModesUrl() {
        return this.modesUrl;
    }

    public int getMusic() {
        return this.music;
    }

    public int getReversal() {
        return this.reversal;
    }

    public int getReversalFun() {
        return this.reversalFun;
    }

    public int getShake() {
        return this.shake;
    }

    public int getSlide() {
        return this.slide;
    }

    public int getSmooth() {
        return this.smooth;
    }

    public int getSpeed() {
        return this.speed;
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

    public int getStrokeType() {
        return this.strokeType;
    }

    public int getSvgLeftRight() {
        return this.svgLeftRight;
    }

    public String getSvgName() {
        return this.svgName;
    }

    public String getSvgUrl() {
        return this.svgUrl;
    }

    public long getSvgVersion() {
        return this.svgVersion;
    }

    public String getType() {
        return this.type;
    }

    public int getVibration() {
        return this.vibration;
    }

    public int getVspeed() {
        return this.vspeed;
    }

    public void setClimax(String str) {
        this.climax = str;
    }

    public void setMode(int i) {
        this.mode = i;
    }

    public void setModeStatus(int i) {
        this.modeStatus = i;
    }

    public void setModes(Modes modes) {
        this.modes = modes;
    }

    public void setModesUrl(Modes modes) {
        this.modesUrl = modes;
    }

    public void setMusic(int i) {
        this.music = i;
    }

    public void setReversal(int i) {
        this.reversal = i;
    }

    public void setReversalFun(int i) {
        this.reversalFun = i;
    }

    public void setShake(int i) {
        this.shake = i;
    }

    public void setSlide(int i) {
        this.slide = i;
    }

    public void setSmooth(int i) {
        this.smooth = i;
    }

    public void setSpeed(int i) {
        this.speed = i;
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

    public void setStrokeType(int i) {
        this.strokeType = i;
    }

    public void setSvgLeftRight(int i) {
        this.svgLeftRight = i;
    }

    public void setSvgName(String str) {
        this.svgName = str;
    }

    public void setSvgUrl(String str) {
        this.svgUrl = str;
    }

    public void setSvgVersion(long j) {
        this.svgVersion = j;
    }

    public void setType(String str) {
        this.type = str;
    }

    public void setVibration(int i) {
        this.vibration = i;
    }

    public void setVspeed(int i) {
        this.vspeed = i;
    }

    public String toString() {
        return "SocketProductModel{type='" + this.type + "', speed=" + this.speed + ", vspeed=" + this.vspeed + ", mode=" + this.mode + ", vibration=" + this.vibration + ", music=" + this.music + ", shake=" + this.shake + ", slide=" + this.slide + ", reversal=" + this.reversal + ", reversalFun=" + this.reversalFun + ", strokeType=" + this.strokeType + ", strokeMin=" + this.strokeMin + ", strokeMax=" + this.strokeMax + ", strokeNow=" + this.strokeNow + ", smooth=" + this.smooth + ", climax='" + this.climax + "', modes=" + this.modes + ", modesUrl=" + this.modesUrl + ", svgName='" + this.svgName + "', svgVersion=" + this.svgVersion + ", svgUrl='" + this.svgUrl + "', svgLeftRight=" + this.svgLeftRight + ", modeStatus=" + this.modeStatus + '}';
    }
}
