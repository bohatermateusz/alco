package com.hismith.hismithapp.utils;

import android.text.TextUtils;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class DataBakUtil {
    private List<Float> mPoint;
    private String mModeData = "1001E004G11F00004G10000002G0010304G12828002G10000001G0040502G0010504G";
    private String[] mModeDataArr = "1001E004G11F00004G10000002G0010304G12828002G10000001G0040502G0010504G".split("G");
    private int intervalTime = 5;
    private int totalPoint = 60;
    private Map<Integer, List<Float>> modeList = new HashMap();

    class ItemModel {
        int cycles;
        float end;
        int repeat;
        float start;
        float timeSecond;

        public ItemModel(int i) {
            this.repeat = i;
        }

        public int getCycles() {
            return this.cycles;
        }

        public float getEnd() {
            return this.end;
        }

        public int getRepeat() {
            return this.repeat;
        }

        public float getStart() {
            return this.start;
        }

        public float getTimeSecond() {
            return this.timeSecond;
        }

        public void setCycles(int i) {
            this.cycles = i;
        }

        public void setEnd(float f) {
            this.end = f;
        }

        public void setRepeat(int i) {
            this.repeat = i;
        }

        public void setStart(float f) {
            this.start = f;
        }

        public void setTimeSecond(float f) {
            this.timeSecond = f;
        }
    }

    private List<Float> getCyclesList(int i, int i2) {
        ArrayList arrayList = new ArrayList();
        for (int i3 = i - 1; i3 < i2; i3++) {
            int i4 = 0;
            boolean equals = "1".equals(this.mModeDataArr[i3].substring(0, 1));
            String[] strArr = this.mModeDataArr;
            if (equals) {
                float intValue = Integer.valueOf(strArr[i3].substring(1, 3), 16).intValue();
                float intValue2 = Integer.valueOf(this.mModeDataArr[i3].substring(3, 5), 16).intValue();
                float intValue3 = (Integer.valueOf(this.mModeDataArr[i3].substring(5, 7), 16).intValue() * 100) + (Integer.valueOf(this.mModeDataArr[i3].substring(7, 8), 16).intValue() * 10.0f);
                ArrayList arrayList2 = new ArrayList();
                if (intValue3 > this.intervalTime) {
                    float ceil = (float) Math.ceil(intValue3 / r7);
                    float f = (intValue2 - intValue) / ceil;
                    while (true) {
                        if (i4 >= ceil) {
                            break;
                        }
                        float ceil2 = (float) Math.ceil((r7 * f) + intValue);
                        if (ceil2 > 100.0f) {
                            ceil2 = 100.0f;
                        }
                        arrayList2.add(Float.valueOf(ceil2));
                        i4++;
                    }
                } else if (intValue == intValue2) {
                    arrayList2.add(Float.valueOf(intValue));
                } else {
                    arrayList2.add(Float.valueOf(intValue));
                    arrayList2.add(Float.valueOf(intValue2));
                }
                arrayList.addAll(arrayList2);
            } else {
                int intValue4 = Integer.valueOf(strArr[i3].substring(1, 3), 16).intValue();
                int intValue5 = Integer.valueOf(this.mModeDataArr[i3].substring(3, 5), 16).intValue();
                int intValue6 = Integer.valueOf(this.mModeDataArr[i3].substring(5, 7), 16).intValue();
                while (i4 < intValue6) {
                    arrayList.addAll(getCyclesList(intValue4, intValue5));
                    i4++;
                }
            }
        }
        return arrayList;
    }

    private List<Float> initPoint1(int i) {
        ArrayList arrayList = new ArrayList();
        int i2 = 0;
        while (true) {
            String[] strArr = this.mModeDataArr;
            if (i2 >= strArr.length) {
                break;
            }
            if (!TextUtils.isEmpty(strArr[i2])) {
                if ("0".equals(this.mModeDataArr[i2].substring(0, 1))) {
                    int intValue = Integer.valueOf(this.mModeDataArr[i2].substring(1, 3), 16).intValue();
                    int intValue2 = Integer.valueOf(this.mModeDataArr[i2].substring(3, 5), 16).intValue();
                    int intValue3 = Integer.valueOf(this.mModeDataArr[i2].substring(5, 7), 16).intValue();
                    for (int i3 = 0; i3 < intValue3 && arrayList.size() < this.totalPoint; i3++) {
                        List<Float> cyclesList = getCyclesList(intValue, intValue2);
                        for (int i4 = 0; i4 < cyclesList.size() && arrayList.size() < this.totalPoint; i4++) {
                            if (i <= 0) {
                                arrayList.add(cyclesList.get(i4));
                            } else {
                                i -= this.intervalTime;
                            }
                        }
                    }
                } else {
                    int i5 = i2 + 1;
                    List<Float> cyclesList2 = getCyclesList(i5, i5);
                    for (int i6 = 0; i6 < cyclesList2.size() && arrayList.size() < this.totalPoint; i6++) {
                        if (i <= 0) {
                            arrayList.add(cyclesList2.get(i6));
                        } else {
                            i -= this.intervalTime;
                        }
                    }
                }
            }
            if (arrayList.size() >= this.totalPoint) {
                break;
            }
            i2++;
        }
        return arrayList;
    }
}
