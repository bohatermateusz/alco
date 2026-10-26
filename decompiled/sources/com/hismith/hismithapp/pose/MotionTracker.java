package com.hismith.hismithapp.pose;

import com.google.mediapipe.tasks.components.containers.Landmark;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import java.util.List;

/* loaded from: classes3.dex */
public class MotionTracker {
    private static final float PRESENCE_THRESHOLD = 0.5f;
    private static final float VISIBILITY_THRESHOLD = 0.5f;
    private OnSpeedChangeListener onSpeedChangeListener;
    public static final float[] landmarkWeights = new float[33];
    private static final String[] LANDMARK_NAMES = {"鼻子", "左眼内眼角", "左眼", "左眼外眼角", "右眼内眼角", "右眼", "右眼外眼角", "左耳", "右耳", "左嘴角", "右嘴角", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕", "左小指", "右小指", "左指尖", "右指尖", "左拇指", "右拇指", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "左脚跟", "右脚跟", "左脚尖", "右脚尖"};
    private List<List<Landmark>> previousWorldLandmarks = null;
    private long previousTimestampMs = -1;
    private float maxSpeedForPercent = 2.0f;

    public interface OnSpeedChangeListener {
        void onSpeedChange(int i);
    }

    public MotionTracker() {
        initDefaultWeights();
    }

    private void initDefaultWeights() {
        System.arraycopy(new float[]{1.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.8f, 1.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.5f, 2.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 0, landmarkWeights, 0, 33);
    }

    public void reset() {
        this.previousWorldLandmarks = null;
        this.previousTimestampMs = -1L;
    }

    public void setOnSpeedChangeListener(OnSpeedChangeListener onSpeedChangeListener) {
        this.onSpeedChangeListener = onSpeedChangeListener;
    }

    public void setWeight(int i, float f) {
        if (i >= 0) {
            float[] fArr = landmarkWeights;
            if (i < fArr.length) {
                fArr[i] = f;
            }
        }
    }

    public String updateAndGetMaxMotionSpeed(PoseLandmarkerResult poseLandmarkerResult, long j) {
        float f;
        int i;
        List<Landmark> list;
        if (poseLandmarkerResult == null) {
            return null;
        }
        List<List<Landmark>> worldLandmarks = poseLandmarkerResult.worldLandmarks();
        if (worldLandmarks == null || worldLandmarks.isEmpty()) {
            return null;
        }
        int i2 = 0;
        List<Landmark> list2 = worldLandmarks.get(0);
        List<List<Landmark>> list3 = this.previousWorldLandmarks;
        if (list3 == null || this.previousTimestampMs == -1) {
            this.previousWorldLandmarks = worldLandmarks;
            this.previousTimestampMs = j;
            return null;
        }
        List<Landmark> list4 = list3.get(0);
        float f2 = (j - this.previousTimestampMs) / 1000.0f;
        float f3 = 0.0f;
        if (f2 <= 0.0f) {
            this.previousWorldLandmarks = worldLandmarks;
            this.previousTimestampMs = j;
            return null;
        }
        float f4 = -1.0f;
        int i3 = -1;
        float f5 = 1.0f;
        float f6 = 0.0f;
        float f7 = 0.0f;
        float f8 = 0.0f;
        float f9 = 0.0f;
        while (i2 < list2.size()) {
            float f10 = landmarkWeights[i2];
            if (f10 == f3) {
                i = i2;
                list = list2;
                f = f3;
            } else {
                Landmark landmark = list2.get(i2);
                Landmark landmark2 = list4.get(i2);
                f = f3;
                float floatValue = landmark.presence().orElse(Float.valueOf(f)).floatValue();
                float floatValue2 = landmark.visibility().orElse(Float.valueOf(f)).floatValue();
                if (floatValue < 0.5f || floatValue2 < 0.5f) {
                    i = i2;
                    list = list2;
                } else {
                    float x = landmark.x() - landmark2.x();
                    float y = landmark.y() - landmark2.y();
                    float z = landmark.z() - landmark2.z();
                    float f11 = (x * x) + (y * y) + (z * z);
                    i = i2;
                    list = list2;
                    float sqrt = (float) Math.sqrt(f11);
                    float f12 = sqrt / f2;
                    float f13 = f12 * f10;
                    if (f13 > f4) {
                        f8 = floatValue2;
                        f7 = sqrt;
                        f6 = f12;
                        f4 = f13;
                        i3 = i;
                        f5 = f10;
                        f9 = floatValue;
                    }
                }
            }
            i2 = i + 1;
            f3 = f;
            list2 = list;
        }
        this.previousWorldLandmarks = worldLandmarks;
        this.previousTimestampMs = j;
        if (i3 < 0) {
            return null;
        }
        int round = (int) Math.round(Math.min(100.0d, (f6 / this.maxSpeedForPercent) * 100.0d));
        OnSpeedChangeListener onSpeedChangeListener = this.onSpeedChangeListener;
        if (onSpeedChangeListener != null) {
            onSpeedChangeListener.onSpeedChange(round);
        }
        return String.format("%s 位移:%.3fm 速度:%.2fm/s 权重:%.2f vis:%.2f pre:%.2f", LANDMARK_NAMES[i3], Float.valueOf(f7), Float.valueOf(f6), Float.valueOf(f5), Float.valueOf(f8), Float.valueOf(f9));
    }
}
