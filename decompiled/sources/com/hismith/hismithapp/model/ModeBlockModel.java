package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class ModeBlockModel extends HttpModel {
    private ModeBlock[] modeBlocks;
    private String requestId;

    public class ModeBlock {
        private String blockData;
        private String blockIcon;
        private int blockId;
        private double blockTime;

        public ModeBlock() {
        }

        public String getBlockData() {
            return this.blockData;
        }

        public String getBlockIcon() {
            return this.blockIcon;
        }

        public int getBlockId() {
            return this.blockId;
        }

        public double getBlockTime() {
            return this.blockTime;
        }

        public void setBlockData(String str) {
            this.blockData = str;
        }

        public void setBlockIcon(String str) {
            this.blockIcon = str;
        }

        public void setBlockId(int i) {
            this.blockId = i;
        }

        public void setBlockTime(double d) {
            this.blockTime = d;
        }
    }

    public ModeBlock[] getModeBlocks() {
        return this.modeBlocks;
    }

    public String getRequestId() {
        return this.requestId;
    }

    public void setModeBlocks(ModeBlock[] modeBlockArr) {
        this.modeBlocks = modeBlockArr;
    }

    public void setRequestId(String str) {
        this.requestId = str;
    }
}
