package com.hismith.hismithapp.model;

import android.text.TextUtils;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.product.Factory;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class ProductModel extends HttpModel {
    private int climax;
    private String code;
    private String command;
    private String deviceType;
    private Fun[] funs;
    private int id;
    private String logo;
    private int music;
    private String name;
    private int reversal;
    private int shake;
    private int slide;
    private int svgLeftRight;
    private String svgName;
    private String svgUrl;
    private int svgVersion;

    public class Fun {
        private String fCommand;
        private String fIcon;
        private int fIconLocation;
        private int fId;
        private String fMode;
        private String fSpeedQuery;
        private String fSpeedSet;
        private String fType;
        private Mode[] modes;

        public Fun() {
        }

        public Mode[] getModes() {
            return this.modes;
        }

        public String getfCommand() {
            return this.fCommand;
        }

        public String getfIcon() {
            return this.fIcon;
        }

        public int getfIconLocation() {
            return this.fIconLocation;
        }

        public int getfId() {
            return this.fId;
        }

        public String getfMode() {
            return this.fMode;
        }

        public String getfSpeedQuery() {
            return this.fSpeedQuery;
        }

        public String getfSpeedSet() {
            return this.fSpeedSet;
        }

        public String getfType() {
            return this.fType;
        }

        public void setModes(Mode[] modeArr) {
            this.modes = modeArr;
        }

        public void setfCommand(String str) {
            this.fCommand = str;
        }

        public void setfIcon(String str) {
            this.fIcon = str;
        }

        public void setfIconLocation(int i) {
            this.fIconLocation = i;
        }

        public void setfId(int i) {
            this.fId = i;
        }

        public void setfMode(String str) {
            this.fMode = str;
        }

        public void setfSpeedQuery(String str) {
            this.fSpeedQuery = str;
        }

        public void setfSpeedSet(String str) {
            this.fSpeedSet = str;
        }

        public void setfType(String str) {
            this.fType = str;
        }

        public String toString() {
            return "Fun{fId=" + this.fId + ", fCommand='" + this.fCommand + "', fSpeedSet='" + this.fSpeedSet + "', fSpeedQuery='" + this.fSpeedQuery + "', fMode='" + this.fMode + "', fIcon='" + this.fIcon + "', fIconLocation=" + this.fIconLocation + ", fType='" + this.fType + "', modes=" + Arrays.toString(this.modes) + '}';
        }
    }

    public class Mode {
        private int locationIcon;
        private String mCommand;
        private String mIcon;

        public Mode() {
        }

        public int getLocationIcon() {
            return this.locationIcon;
        }

        public String getmCommand() {
            return this.mCommand;
        }

        public String getmIcon() {
            return this.mIcon;
        }

        public void setLocationIcon(int i) {
            this.locationIcon = i;
        }

        public void setmCommand(String str) {
            this.mCommand = str;
        }

        public void setmIcon(String str) {
            this.mIcon = str;
        }

        public String toString() {
            return "Mode{mCommand='" + this.mCommand + "', mIcon='" + this.mIcon + "', locationIcon=" + this.locationIcon + '}';
        }
    }

    public int getClimax() {
        return this.climax;
    }

    public String getCode() {
        return this.code;
    }

    public String getCommand() {
        return this.command;
    }

    public String getDeviceType() {
        if (TextUtils.isEmpty(this.deviceType)) {
            this.deviceType = Factory.buildDeviceType(this.funs);
        }
        return this.deviceType;
    }

    public Fun[] getFuns() {
        return this.funs;
    }

    public int getId() {
        return this.id;
    }

    public String getLogo() {
        return this.logo;
    }

    public int getMusic() {
        return this.music;
    }

    public String getName() {
        return this.name;
    }

    public int getReversal() {
        return this.reversal;
    }

    public int getShake() {
        return this.shake;
    }

    public int getSlide() {
        return this.slide;
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

    public int getSvgVersion() {
        return this.svgVersion;
    }

    public void setClimax(int i) {
        this.climax = i;
    }

    public void setCode(String str) {
        this.code = str;
    }

    public void setCommand(String str) {
        this.command = str;
    }

    public void setDeviceType(String str) {
        this.deviceType = str;
    }

    public void setFuns(Fun[] funArr) {
        this.funs = funArr;
    }

    public void setId(int i) {
        this.id = i;
    }

    public void setLogo(String str) {
        this.logo = str;
    }

    public void setMusic(int i) {
        this.music = i;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setReversal(int i) {
        this.reversal = i;
    }

    public void setShake(int i) {
        this.shake = i;
    }

    public void setSlide(int i) {
        this.slide = i;
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

    public void setSvgVersion(int i) {
        this.svgVersion = i;
    }

    public String toString() {
        return "ProductModel{id=" + this.id + ", code='" + this.code + "', command='" + this.command + "', deviceType='" + getDeviceType() + "', name='" + this.name + "', shake=" + this.shake + ", slide=" + this.slide + ", climax=" + this.climax + ", music=" + this.music + ", reversal=" + this.reversal + ", logo='" + this.logo + "', svgName='" + this.svgName + "', svgVersion=" + this.svgVersion + ", svgUrl='" + this.svgUrl + "', svgLeftRight=" + this.svgLeftRight + ", funs=" + Arrays.toString(this.funs) + '}';
    }
}
