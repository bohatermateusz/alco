package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class BarrageModel extends HttpModel {
    private Array[] datas;

    public class Array {
        private String color;
        private String content;
        private double second;

        public Array() {
        }

        public String getColor() {
            return this.color;
        }

        public String getContent() {
            return this.content;
        }

        public double getSecond() {
            return this.second;
        }

        public void setColor(String str) {
            this.color = str;
        }

        public void setContent(String str) {
            this.content = str;
        }

        public void setSecond(double d) {
            this.second = d;
        }

        public String toString() {
            return "Array{second=" + this.second + ", content='" + this.content + "', color='" + this.color + "'}";
        }
    }

    public Array[] getDatas() {
        return this.datas;
    }

    public void setDatas(Array[] arrayArr) {
        this.datas = arrayArr;
    }

    public String toString() {
        return "BarrageModel{datas=" + Arrays.toString(this.datas) + '}';
    }
}
