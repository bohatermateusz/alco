package com.hismith.hismithapp.pose;

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import master.flame.danmaku.danmaku.model.android.DanmakuFactory;

/* loaded from: classes3.dex */
public class PoseSpeedTracker {
    private static final int LEFT_HIP = 23;
    private static final int LEFT_WRIST = 15;
    private static final int NOSE = 0;
    private static final int RIGHT_HIP = 24;
    private static final int RIGHT_WRIST = 16;
    private static final float WEIGHT_HIP = 2.0f;
    private static final float WEIGHT_NOSE = 1.0f;
    private static final float WEIGHT_WRIST = 1.5f;
    private float[] lastFeature;
    private float gain = 30.0f;
    private float deadZone = 0.005f;
    private int smoothWindow = 5;
    private float repeatThreshold = 0.92f;
    private float accumulateRate = 2.0f;
    private int historyDelayMs = 1000;
    private long lastTimestamp = -1;
    private float boostSpeed = 0.0f;
    private final Deque<Float> smoothQueue = new ArrayDeque();
    private final Deque<HistoryFrame> historyQueue = new ArrayDeque();

    private static class HistoryFrame {
        float[] feature;
        long timestamp;

        HistoryFrame(long j, float[] fArr) {
            this.timestamp = j;
            this.feature = fArr;
        }
    }

    private float cosineSimilarity(float[] fArr, float[] fArr2) {
        if (fArr == null || fArr2 == null || fArr.length != fArr2.length) {
            return 0.0f;
        }
        float f = 0.0f;
        float f2 = 0.0f;
        float f3 = 0.0f;
        for (int i = 0; i < fArr.length; i++) {
            float f4 = fArr[i];
            float f5 = fArr2[i];
            f2 += f4 * f5;
            f += f4 * f4;
            f3 += f5 * f5;
        }
        if (f != 0.0f && f3 != 0.0f) {
            return f2 / ((float) (Math.sqrt(f) * Math.sqrt(f3)));
        }
        return 0.0f;
    }

    private float distance(float[] fArr, float[] fArr2, int i) {
        float f = fArr[i] - fArr2[i];
        int i2 = i + 1;
        float f2 = fArr[i2] - fArr2[i2];
        int i3 = i + 2;
        float f3 = fArr[i3] - fArr2[i3];
        return (float) Math.sqrt((f * f) + (f2 * f2) + (f3 * f3));
    }

    private void fillPoint(float[] fArr, int i, NormalizedLandmark normalizedLandmark) {
        fArr[i] = normalizedLandmark.x();
        fArr[i + 1] = normalizedLandmark.y();
        fArr[i + 2] = normalizedLandmark.z();
    }

    public float getAccumulateRate() {
        return this.accumulateRate;
    }

    public float getDeadZone() {
        return this.deadZone;
    }

    public float getGain() {
        return this.gain;
    }

    public int getHistoryDelayMs() {
        return this.historyDelayMs;
    }

    public float getRepeatThreshold() {
        return this.repeatThreshold;
    }

    public int getSmoothWindow() {
        return this.smoothWindow;
    }

    public void reset() {
        this.lastFeature = null;
        this.lastTimestamp = -1L;
        this.boostSpeed = 0.0f;
        this.smoothQueue.clear();
        this.historyQueue.clear();
    }

    public void setAccumulateRate(float f) {
        this.accumulateRate = f;
    }

    public void setDeadZone(float f) {
        this.deadZone = f;
    }

    public void setGain(float f) {
        this.gain = f;
    }

    public void setHistoryDelayMs(int i) {
        this.historyDelayMs = i;
    }

    public void setRepeatThreshold(float f) {
        this.repeatThreshold = f;
    }

    public void setSmoothWindow(int i) {
        this.smoothWindow = Math.max(1, i);
    }

    public int update(List<NormalizedLandmark> list, long j) {
        float f;
        Deque<Float> deque;
        float f2;
        boolean z = false;
        if (list == null || list.size() < 25) {
            return 0;
        }
        float[] fArr = new float[15];
        fillPoint(fArr, 0, list.get(0));
        fillPoint(fArr, 3, list.get(15));
        fillPoint(fArr, 6, list.get(16));
        fillPoint(fArr, 9, list.get(23));
        fillPoint(fArr, 12, list.get(24));
        float[] fArr2 = this.lastFeature;
        if (fArr2 == null) {
            this.lastFeature = (float[]) fArr.clone();
            this.lastTimestamp = j;
            this.historyQueue.addLast(new HistoryFrame(j, (float[]) fArr.clone()));
            return 0;
        }
        long j2 = j - this.lastTimestamp;
        if (j2 <= 0) {
            j2 = 1;
        }
        float distance = (((((distance(fArr, fArr2, 0) * 1.0f) + (distance(fArr, this.lastFeature, 3) * WEIGHT_WRIST)) + (distance(fArr, this.lastFeature, 6) * WEIGHT_WRIST)) + (distance(fArr, this.lastFeature, 9) * 2.0f)) + (distance(fArr, this.lastFeature, 12) * 2.0f)) / 8.0f;
        float f3 = 100.0f;
        if (distance < this.deadZone) {
            f = 0.0f;
        } else {
            f = this.gain * distance;
            if (f > 100.0f) {
                f = 100.0f;
            }
        }
        this.smoothQueue.addLast(Float.valueOf(f));
        while (true) {
            int size = this.smoothQueue.size();
            int i = this.smoothWindow;
            deque = this.smoothQueue;
            if (size <= i) {
                break;
            }
            deque.removeFirst();
        }
        Iterator<Float> it = deque.iterator();
        float f4 = 0.0f;
        while (it.hasNext()) {
            f4 += it.next().floatValue();
        }
        float size2 = f4 / this.smoothQueue.size();
        this.historyQueue.addLast(new HistoryFrame(j, (float[]) fArr.clone()));
        while (!this.historyQueue.isEmpty() && j - this.historyQueue.peekFirst().timestamp > DanmakuFactory.MIN_DANMAKU_DURATION) {
            this.historyQueue.removeFirst();
        }
        Iterator<HistoryFrame> it2 = this.historyQueue.iterator();
        HistoryFrame historyFrame = null;
        long j3 = Long.MAX_VALUE;
        while (it2.hasNext()) {
            HistoryFrame next = it2.next();
            boolean z2 = z;
            Iterator<HistoryFrame> it3 = it2;
            float f5 = f3;
            float f6 = size2;
            long abs = Math.abs((j - next.timestamp) - this.historyDelayMs);
            if (abs < j3) {
                j3 = abs;
                historyFrame = next;
            }
            it2 = it3;
            z = z2;
            f3 = f5;
            size2 = f6;
        }
        boolean z3 = z;
        float f7 = f3;
        float f8 = size2;
        float cosineSimilarity = historyFrame != null ? cosineSimilarity(fArr, historyFrame.feature) : 0.0f;
        if (historyFrame != null && cosineSimilarity >= this.repeatThreshold && distance > this.deadZone * 0.5f) {
            z3 = true;
        }
        this.lastFeature = (float[]) fArr.clone();
        this.lastTimestamp = j;
        float f9 = j2 / 1000.0f;
        if (distance < this.deadZone * 0.5f) {
            float f10 = this.boostSpeed - ((this.accumulateRate * 4.0f) * f9);
            this.boostSpeed = f10;
            if (f10 < 0.0f) {
                f2 = 0.0f;
                this.boostSpeed = 0.0f;
            }
            f2 = 0.0f;
        } else if (z3) {
            float f11 = this.boostSpeed + (this.accumulateRate * f9);
            this.boostSpeed = f11;
            if (f11 > f7) {
                this.boostSpeed = f7;
            }
            f2 = 0.0f;
        } else {
            f2 = 0.0f;
            if (cosineSimilarity > 0.0f && cosineSimilarity < this.repeatThreshold) {
                this.boostSpeed = 0.0f;
            }
        }
        float f12 = f8 + this.boostSpeed;
        float f13 = f12 > 100.0f ? 100.0f : f12;
        return Math.round(f13 < f2 ? f2 : f13);
    }
}
