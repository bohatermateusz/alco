package com.hismith.hismithapp.pose;

import android.util.Log;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;

/* loaded from: classes3.dex */
public class PoseSpeedTracker2 {
    private static final int LEFT_HIP = 23;
    private static final int LEFT_WRIST = 15;
    private static final int NOSE = 0;
    private static final int RIGHT_HIP = 24;
    private static final int RIGHT_WRIST = 16;
    private static final String TAG = "PoseSpeed";
    private float[] lastFeature;
    private float gain = 12.0f;
    private float deadZone = 0.05f;
    private float emaAlpha = 0.3f;
    private int smoothWindow = 3;
    private float repeatThreshold = 0.92f;
    private float accumulateRate = 1.0f;
    private long peakHoldMs = 100;
    private float emaSpeed = 0.0f;
    private float peakSpeed = 0.0f;
    private long peakTimestamp = 0;
    private float boostSpeed = 0.0f;
    private long lastTimestamp = -1;
    private final Deque<Float> smoothQueue = new ArrayDeque();
    private final Deque<HistoryFrame> historyQueue = new ArrayDeque();
    private float prevRawSpeedUnclamped = 0.0f;
    private float accel = 0.0f;
    private float accelLP = 0.0f;
    private float accelLPAlpha = 0.2f;
    private float headWeight = 1.0f;
    private float handWeight = 0.6f;
    private float hipWeight = 1.5f;

    private static class HistoryFrame {
        float[] feature;
        long timestamp;

        HistoryFrame(long j, float[] fArr) {
            this.timestamp = j;
            this.feature = fArr;
        }
    }

    public static class MoveResult {
        public float move;
        public int part;

        public MoveResult(float f, int i) {
            this.move = f;
            this.part = i;
        }
    }

    private float calculateBodyScale(float[] fArr) {
        float f = fArr[9] - fArr[12];
        float f2 = fArr[10] - fArr[13];
        float f3 = fArr[11] - fArr[14];
        return (float) Math.sqrt((f * f) + (f2 * f2) + (f3 * f3));
    }

    private MoveResult calculateMove(float[] fArr, float[] fArr2) {
        int i = 0;
        float distance = distance(fArr, fArr2, 0);
        float max = Math.max(distance(fArr, fArr2, 3), distance(fArr, fArr2, 6));
        float max2 = Math.max(distance(fArr, fArr2, 9), distance(fArr, fArr2, 12));
        float f = distance * this.headWeight;
        float f2 = max * this.handWeight;
        float f3 = max2 * this.hipWeight;
        if (f2 > f) {
            i = 1;
            f = f2;
        }
        if (f3 > f) {
            i = 2;
        } else {
            f3 = f;
        }
        return new MoveResult(f3, i);
    }

    private boolean checkRepeat(float[] fArr, long j) {
        HistoryFrame findHistoryFrame = findHistoryFrame(j - 800);
        return findHistoryFrame != null && cosineSimilarity(normalizePose(fArr), normalizePose(findHistoryFrame.feature)) >= this.repeatThreshold;
    }

    private float cosineSimilarity(float[] fArr, float[] fArr2) {
        if (fArr != null && fArr2 != null && fArr.length == fArr2.length) {
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
                return f2 / (((float) Math.sqrt(f)) * ((float) Math.sqrt(f3)));
            }
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

    private HistoryFrame findHistoryFrame(long j) {
        HistoryFrame historyFrame = null;
        long j2 = Long.MAX_VALUE;
        for (HistoryFrame historyFrame2 : this.historyQueue) {
            long abs = Math.abs(historyFrame2.timestamp - j);
            if (abs < j2) {
                historyFrame = historyFrame2;
                j2 = abs;
            }
        }
        return historyFrame;
    }

    private float[] normalizePose(float[] fArr) {
        float[] fArr2 = (float[]) fArr.clone();
        float f = (fArr[9] + fArr[12]) / 2.0f;
        float f2 = (fArr[10] + fArr[13]) / 2.0f;
        for (int i = 0; i < fArr2.length; i += 3) {
            fArr2[i] = fArr2[i] - f;
            int i2 = i + 1;
            fArr2[i2] = fArr2[i2] - f2;
        }
        float calculateBodyScale = calculateBodyScale(fArr);
        if (calculateBodyScale < 0.01f) {
            calculateBodyScale = 0.01f;
        }
        for (int i3 = 0; i3 < fArr2.length; i3++) {
            fArr2[i3] = fArr2[i3] / calculateBodyScale;
        }
        return fArr2;
    }

    public float getAccumulateRate() {
        return this.accumulateRate;
    }

    public float getDeadZone() {
        return this.deadZone;
    }

    public float getEmaAlpha() {
        return this.emaAlpha;
    }

    public float getGain() {
        return this.gain;
    }

    public long getPeakHoldMs() {
        return this.peakHoldMs;
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
        this.emaSpeed = 0.0f;
        this.peakSpeed = 0.0f;
        this.boostSpeed = 0.0f;
        this.peakTimestamp = 0L;
        this.smoothQueue.clear();
        this.historyQueue.clear();
        this.prevRawSpeedUnclamped = 0.0f;
        this.accel = 0.0f;
        this.accelLP = 0.0f;
    }

    public void setAccumulateRate(float f) {
        this.accumulateRate = f;
    }

    public void setDeadZone(float f) {
        this.deadZone = f;
    }

    public void setEmaAlpha(float f) {
        this.emaAlpha = Math.max(0.05f, Math.min(1.0f, f));
    }

    public void setGain(float f) {
        this.gain = f;
    }

    public void setPeakHoldMs(long j) {
        this.peakHoldMs = j;
    }

    public void setRepeatThreshold(float f) {
        this.repeatThreshold = f;
    }

    public void setSmoothWindow(int i) {
        this.smoothWindow = Math.max(1, i);
    }

    /* JADX WARN: Removed duplicated region for block: B:32:0x0101  */
    /* JADX WARN: Removed duplicated region for block: B:36:0x013f A[LOOP:0: B:34:0x0133->B:36:0x013f, LOOP_END] */
    /* JADX WARN: Removed duplicated region for block: B:37:0x0143 A[EDGE_INSN: B:37:0x0143->B:38:0x0143 BREAK  A[LOOP:0: B:34:0x0133->B:36:0x013f], SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:41:0x014f A[LOOP:1: B:39:0x0149->B:41:0x014f, LOOP_END] */
    /* JADX WARN: Removed duplicated region for block: B:52:0x01a4  */
    /* JADX WARN: Removed duplicated region for block: B:55:0x01ac  */
    /* JADX WARN: Removed duplicated region for block: B:58:0x01a7  */
    /* JADX WARN: Removed duplicated region for block: B:60:0x010c  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public int update(List<NormalizedLandmark> list, long j) {
        float f;
        float f2;
        float f3;
        float f4;
        float f5;
        float f6;
        float f7;
        int size;
        int i;
        Deque<Float> deque;
        Iterator<Float> it;
        float f8;
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
            Log.d(TAG, "init, timestamp=" + j);
            this.prevRawSpeedUnclamped = 0.0f;
            return 0;
        }
        long j2 = j - this.lastTimestamp;
        if (j2 <= 0) {
            j2 = 100;
        }
        MoveResult calculateMove = calculateMove(fArr, fArr2);
        float calculateBodyScale = calculateBodyScale(fArr);
        if (calculateBodyScale < 0.01f) {
            calculateBodyScale = 0.01f;
        }
        float f9 = calculateMove.move / calculateBodyScale;
        float f10 = j2;
        float f11 = (f9 / f10) * 1000.0f * this.gain;
        float f12 = f9 < this.deadZone ? 0.0f : f11;
        if (f12 > 100.0f) {
            f12 = 100.0f;
        }
        if (f12 < 0.0f) {
            f12 = 0.0f;
        }
        this.accel = 0.0f;
        long j3 = j2;
        if (this.lastTimestamp > 0) {
            float f13 = f10 / 1000.0f;
            if (f13 > 0.0f) {
                f = (f11 - this.prevRawSpeedUnclamped) / f13;
                this.accel = f;
                float f14 = this.accelLPAlpha;
                this.accelLP = (f * f14) + ((1.0f - f14) * this.accelLP);
                this.prevRawSpeedUnclamped = f11;
                float f15 = this.emaAlpha;
                f2 = (f15 * f12) + ((1.0f - f15) * this.emaSpeed);
                this.emaSpeed = f2;
                f3 = this.peakSpeed;
                if (f2 <= f3) {
                    this.peakSpeed = f2;
                    this.peakTimestamp = j;
                    f4 = 1.0f;
                    f5 = f12;
                    f6 = 100.0f;
                    f7 = f9;
                } else {
                    f4 = 1.0f;
                    f5 = f12;
                    f6 = 100.0f;
                    f7 = f9;
                    if (j - this.peakTimestamp > this.peakHoldMs) {
                        f3 *= 0.85f;
                        this.peakSpeed = f3;
                        if (f3 < f2) {
                            this.peakSpeed = f2;
                        }
                    }
                    f2 = f3;
                }
                this.smoothQueue.addLast(Float.valueOf(f2));
                while (true) {
                    size = this.smoothQueue.size();
                    i = this.smoothWindow;
                    deque = this.smoothQueue;
                    if (size > i) {
                        break;
                    }
                    deque.removeFirst();
                }
                it = deque.iterator();
                float f16 = 0.0f;
                while (it.hasNext()) {
                    f16 += it.next().floatValue();
                }
                float size2 = f16 / this.smoothQueue.size();
                this.historyQueue.addLast(new HistoryFrame(j, (float[]) fArr.clone()));
                while (!this.historyQueue.isEmpty() && j - this.historyQueue.peekFirst().timestamp > 1500) {
                    this.historyQueue.removeFirst();
                }
                boolean checkRepeat = checkRepeat(fArr, j);
                float f17 = (size2 * f4) + (this.boostSpeed * 0.15f);
                f8 = f17 <= f6 ? f6 : f17;
                if (f8 < 0.0f) {
                    f8 = 0.0f;
                }
                this.lastFeature = (float[]) fArr.clone();
                this.lastTimestamp = j;
                int round = Math.round(f8);
                Log.d(TAG, String.format(Locale.US, "ts=%d | move=%.5f | scale=%.4f | normMove=%.5f | dt=%d | rawUnclamped=%.2f | rawClipped=%.2f | ema=%.2f | peak=%.2f | smooth=%.2f | boost=%.2f | accel=%.2f | accelLP=%.2f | repeated=%b | result=%d", Long.valueOf(j), Float.valueOf(calculateMove.move), Float.valueOf(calculateBodyScale), Float.valueOf(f7), Long.valueOf(j3), Float.valueOf(f11), Float.valueOf(f5), Float.valueOf(this.emaSpeed), Float.valueOf(this.peakSpeed), Float.valueOf(size2), Float.valueOf(this.boostSpeed), Float.valueOf(this.accel), Float.valueOf(this.accelLP), Boolean.valueOf(checkRepeat), Integer.valueOf(round)));
                return round;
            }
        }
        f = 0.0f;
        float f142 = this.accelLPAlpha;
        this.accelLP = (f * f142) + ((1.0f - f142) * this.accelLP);
        this.prevRawSpeedUnclamped = f11;
        float f152 = this.emaAlpha;
        f2 = (f152 * f12) + ((1.0f - f152) * this.emaSpeed);
        this.emaSpeed = f2;
        f3 = this.peakSpeed;
        if (f2 <= f3) {
        }
        this.smoothQueue.addLast(Float.valueOf(f2));
        while (true) {
            size = this.smoothQueue.size();
            i = this.smoothWindow;
            deque = this.smoothQueue;
            if (size > i) {
            }
            deque.removeFirst();
        }
        it = deque.iterator();
        float f162 = 0.0f;
        while (it.hasNext()) {
        }
        float size22 = f162 / this.smoothQueue.size();
        this.historyQueue.addLast(new HistoryFrame(j, (float[]) fArr.clone()));
        while (!this.historyQueue.isEmpty()) {
            this.historyQueue.removeFirst();
        }
        boolean checkRepeat2 = checkRepeat(fArr, j);
        float f172 = (size22 * f4) + (this.boostSpeed * 0.15f);
        if (f172 <= f6) {
        }
        if (f8 < 0.0f) {
        }
        this.lastFeature = (float[]) fArr.clone();
        this.lastTimestamp = j;
        int round2 = Math.round(f8);
        Log.d(TAG, String.format(Locale.US, "ts=%d | move=%.5f | scale=%.4f | normMove=%.5f | dt=%d | rawUnclamped=%.2f | rawClipped=%.2f | ema=%.2f | peak=%.2f | smooth=%.2f | boost=%.2f | accel=%.2f | accelLP=%.2f | repeated=%b | result=%d", Long.valueOf(j), Float.valueOf(calculateMove.move), Float.valueOf(calculateBodyScale), Float.valueOf(f7), Long.valueOf(j3), Float.valueOf(f11), Float.valueOf(f5), Float.valueOf(this.emaSpeed), Float.valueOf(this.peakSpeed), Float.valueOf(size22), Float.valueOf(this.boostSpeed), Float.valueOf(this.accel), Float.valueOf(this.accelLP), Boolean.valueOf(checkRepeat2), Integer.valueOf(round2)));
        return round2;
    }
}
