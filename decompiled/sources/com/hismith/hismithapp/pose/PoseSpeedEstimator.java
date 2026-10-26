package com.hismith.hismithapp.pose;

import android.util.Log;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import com.hismith.hismithapp.pose.PoseSpeedEstimator;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/* loaded from: classes3.dex */
public class PoseSpeedEstimator {
    private static final float EMA_ALPHA = 0.3f;
    private static final int HISTORY_SIZE = 15;
    private static final int LEFT_HIP = 23;
    private static final int LEFT_WRIST = 15;
    private static final float MAX_SINGLE_SPEED = 2.0f;
    private static final float MIN_VELOCITY = 0.002f;
    private static final int NOSE = 0;
    private static final float OUTLIER_THRESHOLD = 3.0f;
    private static final int RIGHT_HIP = 24;
    private static final int RIGHT_WRIST = 16;
    private static final String TAG = "PoseSpeedEstimator";
    private static final float WEIGHT_HIP = 0.5f;
    private static final float WEIGHT_NOSE = 0.3f;
    private static final float WEIGHT_WRIST = 0.2f;
    private SpeedListener listener;
    private LinkedList<FrameData> history = new LinkedList<>();
    private float smoothedSpeed = 0.0f;
    private float lastValidSpeed = 0.0f;
    private long lastTimestamp = -1;
    private int totalFrames = 0;
    private int discardedFrames = 0;
    private float maxSpeedFor100 = 0.5f;

    /* JADX INFO: Access modifiers changed from: private */
    static class FrameData {
        float[] hipCenter;
        float[] leftHip;
        float[] leftWrist;
        float[] nose;
        float[] rightHip;
        float[] rightWrist;
        long timestamp;

        private FrameData() {
        }

        float[] getHipCenter() {
            return this.hipCenter;
        }

        float[] getLeftWrist() {
            return this.leftWrist;
        }

        float[] getNose() {
            return this.nose;
        }

        float[] getRightWrist() {
            return this.rightWrist;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    interface PointExtractor {
        float[] extract(FrameData frameData);
    }

    public interface SpeedListener {
        void onSpeedUpdated(float f, int i);
    }

    public PoseSpeedEstimator(SpeedListener speedListener) {
        this.listener = speedListener;
    }

    private float[] averagePoint(float[] fArr, float[] fArr2) {
        if (fArr == null && fArr2 == null) {
            return null;
        }
        return fArr == null ? fArr2 : fArr2 == null ? fArr : new float[]{(fArr[0] + fArr2[0]) / 2.0f, (fArr[1] + fArr2[1]) / 2.0f, (fArr[2] + fArr2[2]) / 2.0f};
    }

    private float[] getPoint(NormalizedLandmark normalizedLandmark) {
        if (normalizedLandmark == null || !isReliable(normalizedLandmark)) {
            return null;
        }
        return new float[]{normalizedLandmark.x(), normalizedLandmark.y(), normalizedLandmark.z()};
    }

    private void handleNoPerson(String str) {
        Log.d(TAG, "无有效人体: " + str + ", 速度衰减");
        float f = this.lastValidSpeed * 0.7f;
        if (f < 0.01f) {
            f = 0.0f;
        }
        Log.d(TAG, "衰减后速度=" + f);
        notifySpeed(f);
    }

    private boolean isReliable(NormalizedLandmark normalizedLandmark) {
        return normalizedLandmark != null && normalizedLandmark.visibility().orElse(Float.valueOf(0.0f)).floatValue() >= 0.6f && normalizedLandmark.presence().orElse(Float.valueOf(0.0f)).floatValue() >= 0.5f;
    }

    private float norm(float[] fArr) {
        float f = fArr[0];
        float f2 = fArr[1];
        float f3 = fArr[2];
        return (float) Math.sqrt((f * f) + (f2 * f2) + (f3 * f3));
    }

    private void notifySpeed(float f) {
        SpeedListener speedListener = this.listener;
        if (speedListener != null) {
            speedListener.onSpeedUpdated(f, (int) Math.min(100.0f, (f / this.maxSpeedFor100) * 100.0f));
        }
    }

    private void notifySpeed(float f, int i) {
        SpeedListener speedListener = this.listener;
        if (speedListener != null) {
            speedListener.onSpeedUpdated(f, i);
        }
    }

    private float regressionSpeed(LinkedList<FrameData> linkedList, PointExtractor pointExtractor) {
        int size = linkedList.size();
        long j = linkedList.getFirst().timestamp;
        Iterator<FrameData> it = linkedList.iterator();
        double d = 0.0d;
        double d2 = 0.0d;
        double d3 = 0.0d;
        double d4 = 0.0d;
        double d5 = 0.0d;
        double d6 = 0.0d;
        double d7 = 0.0d;
        double d8 = 0.0d;
        while (it.hasNext()) {
            long j2 = j;
            float[] extract = pointExtractor.extract(it.next());
            if (extract == null) {
                return -1.0f;
            }
            double d9 = (r1.timestamp - j2) / 1000.0d;
            d2 += d9;
            float f = extract[0];
            Iterator<FrameData> it2 = it;
            d4 += f;
            float f2 = extract[1];
            d6 += f2;
            float f3 = extract[2];
            d8 += f3;
            double d10 = d + (d9 * d9);
            d3 += f * d9;
            d5 += f2 * d9;
            d7 += f3 * d9;
            j = j2;
            it = it2;
            d = d10;
        }
        double d11 = size;
        double d12 = (d11 * d) - (d2 * d2);
        if (Math.abs(d12) < 1.0E-9d) {
            return 0.0f;
        }
        double d13 = ((d3 * d11) - (d4 * d2)) / d12;
        double d14 = ((d5 * d11) - (d6 * d2)) / d12;
        double d15 = ((d11 * d7) - (d2 * d8)) / d12;
        return (float) Math.sqrt((d13 * d13) + (d14 * d14) + (d15 * d15));
    }

    private float[] subtract(float[] fArr, float[] fArr2) {
        return new float[]{fArr[0] - fArr2[0], fArr[1] - fArr2[1], fArr[2] - fArr2[2]};
    }

    public void logStats() {
        Log.d(TAG, "总帧数: " + this.totalFrames + ", 丢弃帧: " + this.discardedFrames);
    }

    public void reset() {
        this.history.clear();
        this.smoothedSpeed = 0.0f;
        this.lastValidSpeed = 0.0f;
        this.lastTimestamp = -1L;
        this.totalFrames = 0;
        this.discardedFrames = 0;
    }

    public void setMaxSpeedFor100(float f) {
        this.maxSpeedFor100 = f;
    }

    /* JADX WARN: Removed duplicated region for block: B:70:0x02f2  */
    /* JADX WARN: Removed duplicated region for block: B:73:0x02ff  */
    /* JADX WARN: Removed duplicated region for block: B:76:0x030f  */
    /* JADX WARN: Removed duplicated region for block: B:79:0x033c  */
    /* JADX WARN: Removed duplicated region for block: B:82:0x0342  */
    /* JADX WARN: Removed duplicated region for block: B:83:0x0312  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void update(PoseLandmarkerResult poseLandmarkerResult, long j) {
        float f;
        float f2;
        float f3;
        float f4;
        float f5;
        this.totalFrames++;
        Log.d(TAG, "===== 帧开始 =====");
        Log.d(TAG, "timestamp=" + j + ", totalFrames=" + this.totalFrames + ", historySize=" + this.history.size());
        if (poseLandmarkerResult == null || poseLandmarkerResult.landmarks().isEmpty()) {
            handleNoPerson("result null or empty");
            return;
        }
        List<NormalizedLandmark> list = poseLandmarkerResult.landmarks().get(0);
        if (list == null || list.size() < 33) {
            handleNoPerson("landmarks size < 33");
            return;
        }
        if (!isReliable(list.get(23)) && !isReliable(list.get(24))) {
            handleNoPerson("both hips unreliable");
            return;
        }
        FrameData frameData = new FrameData();
        frameData.timestamp = j;
        frameData.nose = getPoint(list.get(0));
        frameData.leftWrist = getPoint(list.get(15));
        frameData.rightWrist = getPoint(list.get(16));
        frameData.leftHip = getPoint(list.get(23));
        frameData.rightHip = getPoint(list.get(24));
        frameData.hipCenter = averagePoint(frameData.leftHip, frameData.rightHip);
        this.history.add(frameData);
        if (this.history.size() > 15) {
            this.history.pollFirst();
        }
        if (this.history.size() < 5) {
            Log.d(TAG, "历史帧数不足5，输出0");
            notifySpeed(0.0f);
            this.lastTimestamp = j;
            return;
        }
        float regressionSpeed = regressionSpeed(this.history, new PointExtractor() { // from class: com.hismith.hismithapp.pose.PoseSpeedEstimator$$ExternalSyntheticLambda0
            @Override // com.hismith.hismithapp.pose.PoseSpeedEstimator.PointExtractor
            public final float[] extract(PoseSpeedEstimator.FrameData frameData2) {
                return frameData2.getHipCenter();
            }
        });
        float regressionSpeed2 = regressionSpeed(this.history, new PointExtractor() { // from class: com.hismith.hismithapp.pose.PoseSpeedEstimator$$ExternalSyntheticLambda1
            @Override // com.hismith.hismithapp.pose.PoseSpeedEstimator.PointExtractor
            public final float[] extract(PoseSpeedEstimator.FrameData frameData2) {
                return frameData2.getNose();
            }
        });
        float regressionSpeed3 = regressionSpeed(this.history, new PointExtractor() { // from class: com.hismith.hismithapp.pose.PoseSpeedEstimator$$ExternalSyntheticLambda2
            @Override // com.hismith.hismithapp.pose.PoseSpeedEstimator.PointExtractor
            public final float[] extract(PoseSpeedEstimator.FrameData frameData2) {
                return frameData2.getLeftWrist();
            }
        });
        float regressionSpeed4 = regressionSpeed(this.history, new PointExtractor() { // from class: com.hismith.hismithapp.pose.PoseSpeedEstimator$$ExternalSyntheticLambda3
            @Override // com.hismith.hismithapp.pose.PoseSpeedEstimator.PointExtractor
            public final float[] extract(PoseSpeedEstimator.FrameData frameData2) {
                return frameData2.getRightWrist();
            }
        });
        Log.d(TAG, String.format("回归速度: hip=%.4f, nose=%.4f, lWrist=%.4f, rWrist=%.4f", Float.valueOf(regressionSpeed), Float.valueOf(regressionSpeed2), Float.valueOf(regressionSpeed3), Float.valueOf(regressionSpeed4)));
        ArrayList arrayList = new ArrayList();
        ArrayList arrayList2 = new ArrayList();
        if (regressionSpeed >= 0.0f) {
            arrayList.add(Float.valueOf(regressionSpeed));
            arrayList2.add("hip");
        }
        if (regressionSpeed2 >= 0.0f) {
            arrayList.add(Float.valueOf(regressionSpeed2));
            arrayList2.add("nose");
        }
        if (regressionSpeed3 >= 0.0f) {
            arrayList.add(Float.valueOf(regressionSpeed3));
            arrayList2.add("lWrist");
        }
        if (regressionSpeed4 >= 0.0f) {
            arrayList.add(Float.valueOf(regressionSpeed4));
            arrayList2.add("rWrist");
        }
        Iterator it = arrayList.iterator();
        float f6 = 0.0f;
        while (it.hasNext()) {
            f6 += ((Float) it.next()).floatValue();
        }
        float size = f6 / arrayList.size();
        Iterator it2 = arrayList.iterator();
        float f7 = 0.0f;
        while (it2.hasNext()) {
            float floatValue = ((Float) it2.next()).floatValue() - size;
            f7 += floatValue * floatValue;
        }
        float sqrt = (float) Math.sqrt(f7 / arrayList.size());
        Log.d(TAG, String.format("速度均值=%.4f, 标准差=%.4f", Float.valueOf(size), Float.valueOf(sqrt)));
        float[] fArr = {0.5f, 0.3f, 0.1f, 0.1f};
        float[] fArr2 = {regressionSpeed, regressionSpeed2, regressionSpeed3, regressionSpeed4};
        String[] strArr = {"hip", "nose", "lWrist", "rWrist"};
        StringBuilder sb = new StringBuilder("异常剔除: ");
        int i = 0;
        float f8 = 0.0f;
        float f9 = 0.0f;
        for (int i2 = 4; i < i2; i2 = 4) {
            float f10 = fArr2[i];
            if (f10 >= 0.0f) {
                if (Math.abs(f10 - size) <= OUTLIER_THRESHOLD * sqrt || sqrt <= 0.01f) {
                    float f11 = fArr[i];
                    f9 += f10 * f11;
                    f8 += f11;
                } else {
                    sb.append(strArr[i]).append("(").append(f10).append(") 剔除 ");
                }
            }
            i++;
        }
        float f12 = f8 > 0.0f ? f9 / f8 : 0.0f;
        Log.d(TAG, sb.toString());
        Log.d(TAG, String.format("加权综合速度 (finalSpeed) = %.4f, 总权重=%.2f", Float.valueOf(f12), Float.valueOf(f8)));
        if (this.history.size() >= 2) {
            LinkedList<FrameData> linkedList = this.history;
            FrameData frameData2 = linkedList.get(linkedList.size() - 2);
            FrameData last = this.history.getLast();
            float f13 = (last.timestamp - frameData2.timestamp) / 1000.0f;
            if (f13 > 0.001f) {
                f = norm(subtract(last.hipCenter, frameData2.hipCenter)) / f13;
                Log.d(TAG, "瞬时速度(inst)=" + f);
                f2 = (f12 * 0.8f) + (f * 0.2f);
                Log.d(TAG, "综合前(combined)=" + f2);
                if (f2 > 2.0f) {
                    Log.d(TAG, "速度超限，截断至 2.0");
                    f2 = 2.0f;
                }
                if (f2 < 0.002f) {
                    Log.d(TAG, "速度小于最小值，置零");
                    f2 = 0.0f;
                }
                f3 = this.smoothedSpeed;
                if (f3 != 0.0f) {
                    this.smoothedSpeed = f2;
                } else {
                    this.smoothedSpeed = (f2 * 0.3f) + (f3 * 0.7f);
                }
                Log.d(TAG, String.format("平滑前=%.4f, 平滑后=%.4f (alpha=%.2f)", Float.valueOf(f2), Float.valueOf(this.smoothedSpeed), Float.valueOf(0.3f)));
                f4 = this.smoothedSpeed;
                if (f4 >= 0.001f) {
                    this.smoothedSpeed = 0.0f;
                    f5 = 0.0f;
                } else {
                    f5 = f4;
                }
                this.lastValidSpeed = f5;
                int min = (int) Math.min(100.0f, (f5 / this.maxSpeedFor100) * 100.0f);
                Log.d(TAG, String.format("最终速度指数: %d (平滑速度=%.4f, maxSpeedFor100=%.2f)", Integer.valueOf(min), Float.valueOf(this.smoothedSpeed), Float.valueOf(this.maxSpeedFor100)));
                Log.d(TAG, "===== 帧结束 =====");
                notifySpeed(this.smoothedSpeed, min);
                this.lastTimestamp = j;
            }
        }
        f = 0.0f;
        Log.d(TAG, "瞬时速度(inst)=" + f);
        f2 = (f12 * 0.8f) + (f * 0.2f);
        Log.d(TAG, "综合前(combined)=" + f2);
        if (f2 > 2.0f) {
        }
        if (f2 < 0.002f) {
        }
        f3 = this.smoothedSpeed;
        if (f3 != 0.0f) {
        }
        Log.d(TAG, String.format("平滑前=%.4f, 平滑后=%.4f (alpha=%.2f)", Float.valueOf(f2), Float.valueOf(this.smoothedSpeed), Float.valueOf(0.3f)));
        f4 = this.smoothedSpeed;
        if (f4 >= 0.001f) {
        }
        this.lastValidSpeed = f5;
        int min2 = (int) Math.min(100.0f, (f5 / this.maxSpeedFor100) * 100.0f);
        Log.d(TAG, String.format("最终速度指数: %d (平滑速度=%.4f, maxSpeedFor100=%.2f)", Integer.valueOf(min2), Float.valueOf(this.smoothedSpeed), Float.valueOf(this.maxSpeedFor100)));
        Log.d(TAG, "===== 帧结束 =====");
        notifySpeed(this.smoothedSpeed, min2);
        this.lastTimestamp = j;
    }
}
