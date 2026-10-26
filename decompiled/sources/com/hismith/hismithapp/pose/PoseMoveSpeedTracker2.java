package com.hismith.hismithapp.pose;

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;

/* loaded from: classes3.dex */
public class PoseMoveSpeedTracker2 {
    private static final int ARM_INIT_SAMPLES = 5;
    private static final float ARM_LEN_ABS_MAX = 2.0f;
    private static final float ARM_LEN_ABS_MIN = 0.5f;
    private static final float ARM_LEN_EMA_ALPHA = 0.2f;
    private static final float ARM_LEN_MAX_RATIO = 1.7f;
    private static final float ARM_LEN_MIN_RATIO = 0.4f;
    private static final float ARM_LEN_REF_MIN = 0.8f;
    private static final int HIP_BOTH = 0;
    private static final int HIP_LEFT = 1;
    private static final int HIP_NONE = -1;
    private static final int HIP_RIGHT = 2;
    private static final int LEFT_HIP = 23;
    private static final int LEFT_SHOULDER = 11;
    private static final int LEFT_WRIST = 15;
    private static final float MAX_HAND_MOVE_PER_FRAME = 0.9f;
    private static final float MIN_HAND_VISIBILITY = 0.65f;
    private static final long NEW_VIDEO_GAP_MS = 5000;
    private static final int NOSE = 0;
    private static final int PART_HAND = 1;
    private static final int PART_HEAD = 0;
    private static final int PART_HIP = 2;
    private static final float PART_INTENSITY_MAX = 2.5f;
    private static final float PART_REF_DEFAULT = 3.0f;
    private static final float PART_REF_EMA_ALPHA = 0.1f;
    private static final float PART_REF_MAX = 10.0f;
    private static final float PART_REF_MIN = 0.2f;
    private static final float PART_REF_PERCENTILE = 0.9f;
    private static final int PART_RING_SIZE = 300;
    private static final int POINT_COUNT = 7;
    private static final int P_LEFT_HIP = 3;
    private static final int P_LEFT_SHOULDER = 5;
    private static final int P_LEFT_WRIST = 1;
    private static final int P_NOSE = 0;
    private static final int P_RIGHT_HIP = 4;
    private static final int P_RIGHT_SHOULDER = 6;
    private static final int P_RIGHT_WRIST = 2;
    private static final int RIGHT_HIP = 24;
    private static final int RIGHT_SHOULDER = 12;
    private static final int RIGHT_WRIST = 16;
    private static final float SHOT_CUT_THRESHOLD = 0.8f;
    private static final String TAG = "PoseMoveSpeedTracker2";
    private boolean hasLastFrame;
    private final Deque<HistoryFrame> history;
    private long lastSeenTimestamp;
    private long lastTimestamp;
    private final float[] partRefVelocity;
    private final List<Deque<Float>> partVelocityRings;
    private float scaleSmooth;
    private float smoothedSpeed;
    private float minVisibility = 0.5f;
    private int maxHistorySize = 6;
    private long maxFrameGapMs = 300;
    private float deadZone = 0.15f;
    private float maxFrameMove = 1.5f;
    private float attackAlpha = 0.55f;
    private float releaseAlpha = 0.45f;
    private float missingDecay = 0.75f;
    private float headWeight = 1.2f;
    private float hipWeight = 1.8f;
    private float handWeight = 0.6f;
    private boolean enableHead = true;
    private boolean enableHand = true;
    private boolean enableHip = true;
    private final float[] handArmLengthRef = {-1.0f, -1.0f};
    private final float[][] handArmLenInit = {new float[5], new float[5]};
    private final int[] handArmLenInitCount = {0, 0};
    private float priorityScale = 70.0f;

    private static class HistoryFrame {
        final float bodyScale;
        final float[] pts;
        final long timestamp;
        final float[] visibility;

        HistoryFrame(long j, float[] fArr, float[] fArr2, float f) {
            this.timestamp = j;
            this.pts = (float[]) fArr.clone();
            this.visibility = (float[]) fArr2.clone();
            this.bodyScale = f;
        }

        public String toString() {
            return "HistoryFrame{timestamp=" + this.timestamp + ", pts=" + Arrays.toString(this.pts) + ", visibility=" + Arrays.toString(this.visibility) + ", bodyScale=" + this.bodyScale + '}';
        }
    }

    public static class MoveResult {
        public float handIntensity;
        public float handMove;
        public float headIntensity;
        public float headMove;
        public float hipIntensity;
        public float hipMove;
        public float move;
        public int part;

        public MoveResult(int i, float f, float f2, float f3, float f4) {
            this.part = i;
            this.move = f;
            this.headMove = f2;
            this.handMove = f3;
            this.hipMove = f4;
        }
    }

    public PoseMoveSpeedTracker2() {
        ArrayList arrayList = new ArrayList();
        this.partVelocityRings = arrayList;
        this.partRefVelocity = new float[]{PART_REF_DEFAULT, PART_REF_DEFAULT, PART_REF_DEFAULT};
        this.history = new ArrayDeque();
        this.lastTimestamp = -1L;
        this.lastSeenTimestamp = -1L;
        this.hasLastFrame = false;
        this.smoothedSpeed = 0.0f;
        this.scaleSmooth = -1.0f;
        arrayList.add(new ArrayDeque());
        arrayList.add(new ArrayDeque());
        arrayList.add(new ArrayDeque());
    }

    /* JADX WARN: Code restructure failed: missing block: B:54:0x00c9, code lost:
    
        if (r5 >= 0.05f) goto L40;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private float calculateBodyScale(float[] fArr, float[] fArr2) {
        float f;
        float f2;
        int i;
        char c;
        float f3;
        float f4;
        int i2;
        char c2;
        char c3;
        char c4;
        char c5;
        char c6;
        char c7;
        float sqrt;
        float f5 = fArr2[5];
        float f6 = this.minVisibility;
        boolean z = f5 >= f6;
        boolean z2 = fArr2[6] >= f6;
        boolean z3 = fArr2[3] >= f6;
        boolean z4 = fArr2[4] >= f6;
        if ((z || z2) && (z3 || z4)) {
            if (z) {
                f = fArr[10] + 0.0f;
                f2 = fArr[11] + 0.0f;
                i = 1;
            } else {
                f = 0.0f;
                f2 = 0.0f;
                i = 0;
            }
            if (z2) {
                f += fArr[12];
                f2 += fArr[13];
                i++;
            }
            int i3 = i;
            c = 6;
            if (z3) {
                float f7 = fArr[6] + 0.0f;
                f4 = fArr[7] + 0.0f;
                f3 = f7;
                i2 = 1;
            } else {
                f3 = 0.0f;
                f4 = 0.0f;
                i2 = 0;
            }
            if (z4) {
                f3 += fArr[8];
                f4 += fArr[9];
                i2++;
            }
            c2 = '\t';
            float f8 = i3;
            float f9 = i2;
            c3 = '\b';
            float f10 = (f / f8) - (f3 / f9);
            c4 = 7;
            c5 = '\r';
            float f11 = (f2 / f8) - (f4 / f9);
            c6 = '\f';
            c7 = 11;
            sqrt = (float) Math.sqrt(Math.pow(f10, 2.0d) + Math.pow(f11, 2.0d));
        } else {
            c = 6;
            c2 = '\t';
            c3 = '\b';
            c4 = 7;
            c5 = '\r';
            c6 = '\f';
            c7 = 11;
        }
        sqrt = -1.0f;
        if (sqrt < 0.0f && z3 && z4) {
            float sqrt2 = (float) Math.sqrt(Math.pow(fArr[c] - fArr[c3], 2.0d) + Math.pow(fArr[c4] - fArr[c2], 2.0d));
            if (sqrt2 >= 0.05f) {
                sqrt = sqrt2;
            }
        }
        if (sqrt < 0.0f && z && z2) {
            float sqrt3 = (float) Math.sqrt(Math.pow(fArr[10] - fArr[c6], 2.0d) + Math.pow(fArr[c7] - fArr[c5], 2.0d));
            if (sqrt3 >= 0.05f) {
                sqrt = sqrt3;
            }
        }
        if (sqrt < 0.05f) {
            sqrt = 0.5f;
        }
        float f12 = this.scaleSmooth;
        if (f12 < 0.0f) {
            this.scaleSmooth = sqrt;
        } else {
            sqrt = (sqrt * 0.25f) + (f12 * 0.75f);
            this.scaleSmooth = sqrt;
        }
        return Math.max(sqrt, 0.05f);
    }

    private MoveResult calculateWindowVelocity() {
        int i;
        float f;
        HistoryFrame historyFrame;
        HistoryFrame historyFrame2;
        double d;
        float[] fArr;
        int i2;
        double d2;
        float[] fArr2;
        double d3;
        ArrayList arrayList = new ArrayList(this.history);
        int i3 = 2;
        if (arrayList.size() < 2) {
            return new MoveResult(0, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        double d4 = 0.0d;
        double d5 = 0.0d;
        int i4 = 1;
        double d6 = 0.0d;
        double d7 = 0.0d;
        while (i4 < arrayList.size()) {
            HistoryFrame historyFrame3 = (HistoryFrame) arrayList.get(i4 - 1);
            HistoryFrame historyFrame4 = (HistoryFrame) arrayList.get(i4);
            int i5 = i3;
            long j = historyFrame4.timestamp - historyFrame3.timestamp;
            if (j > 0) {
                double d8 = (j / 1000.0d) + d4;
                double max = Math.max(historyFrame3.bodyScale, historyFrame4.bodyScale);
                if (max < 0.01d) {
                    max = 0.01d;
                }
                if (!this.enableHead || historyFrame4.visibility[0] < this.minVisibility || historyFrame3.visibility[0] < this.minVisibility) {
                    historyFrame = historyFrame3;
                    historyFrame2 = historyFrame4;
                    d = max;
                } else {
                    historyFrame = historyFrame3;
                    historyFrame2 = historyFrame4;
                    d = max;
                    d5 += cappedDistance(historyFrame4.pts, historyFrame3.pts, 0, d);
                }
                double d9 = d5;
                float[] hipReference = hipReference(historyFrame2.pts, historyFrame2.visibility);
                float[] hipReference2 = hipReference(historyFrame.pts, historyFrame.visibility);
                if (hipReference != null && hipReference2 != null && hipReference[i5] == hipReference2[i5]) {
                    if (this.enableHip) {
                        d6 += cappedPointDistance(hipReference, hipReference2, d);
                    }
                    if (this.enableHand) {
                        int i6 = i5;
                        int i7 = 1;
                        int[] iArr = {1, i6};
                        double d10 = -1.0d;
                        int i8 = 0;
                        while (i8 < i6) {
                            int i9 = iArr[i8];
                            char c = i9 == i7 ? (char) 0 : (char) 1;
                            if (historyFrame2.visibility[i9] < MIN_HAND_VISIBILITY || historyFrame.visibility[i9] < MIN_HAND_VISIBILITY) {
                                fArr = hipReference;
                                i2 = i8;
                                d2 = d;
                                fArr2 = hipReference2;
                            } else {
                                double d11 = d;
                                float[] fArr3 = hipReference;
                                d2 = d11;
                                i2 = i8;
                                fArr2 = hipReference2;
                                double distanceRelative = distanceRelative(historyFrame2.pts, historyFrame.pts, i9, fArr3, fArr2, d2);
                                if (distanceRelative > 0.8999999761581421d) {
                                    fArr = fArr3;
                                } else {
                                    int i10 = i9 == 1 ? 5 : 6;
                                    if (historyFrame2.visibility[i10] >= this.minVisibility) {
                                        int i11 = i10 * 2;
                                        int i12 = i9 * 2;
                                        fArr = fArr3;
                                        float dist2 = dist2(historyFrame2.pts[i11], historyFrame2.pts[i11 + 1], historyFrame2.pts[i12], historyFrame2.pts[i12 + 1]) / ((float) d2);
                                        if (dist2 >= 0.5f && dist2 <= 2.0f) {
                                            float[] fArr4 = this.handArmLengthRef;
                                            float f2 = fArr4[c];
                                            if (f2 <= 0.0f) {
                                                int[] iArr2 = this.handArmLenInitCount;
                                                int i13 = iArr2[c];
                                                if (i13 < 5) {
                                                    float[] fArr5 = this.handArmLenInit[c];
                                                    int i14 = i13 + 1;
                                                    iArr2[c] = i14;
                                                    fArr5[i13] = dist2;
                                                    if (i14 == 5) {
                                                        fArr4[c] = Math.max(median(fArr5), 0.8f);
                                                    }
                                                }
                                            } else if (dist2 >= ARM_LEN_MIN_RATIO * f2 && dist2 <= ARM_LEN_MAX_RATIO * f2) {
                                                fArr4[c] = Math.max((f2 * 0.8f) + (0.2f * dist2), 0.8f);
                                            }
                                        }
                                    } else {
                                        fArr = fArr3;
                                    }
                                    d3 = d10;
                                    if (distanceRelative > d3) {
                                        d10 = distanceRelative;
                                        hipReference2 = fArr2;
                                        d = d2;
                                        i7 = 1;
                                        i6 = 2;
                                        i8 = i2 + 1;
                                        hipReference = fArr;
                                    }
                                    d10 = d3;
                                    hipReference2 = fArr2;
                                    d = d2;
                                    i7 = 1;
                                    i6 = 2;
                                    i8 = i2 + 1;
                                    hipReference = fArr;
                                }
                            }
                            d3 = d10;
                            d10 = d3;
                            hipReference2 = fArr2;
                            d = d2;
                            i7 = 1;
                            i6 = 2;
                            i8 = i2 + 1;
                            hipReference = fArr;
                        }
                        double d12 = d10;
                        if (d12 >= 0.0d && arrayList.size() >= 3) {
                            d7 += Math.min(d12, this.maxFrameMove);
                        }
                    }
                }
                d4 = d8;
                d5 = d9;
            }
            i4++;
            i3 = 2;
        }
        if (d4 <= 0.0d) {
            return new MoveResult(0, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        float f3 = (float) (d5 / d4);
        float f4 = (float) (d6 / d4);
        float f5 = (float) (d7 / d4);
        float normalizePartVelocity = normalizePartVelocity(0, f3);
        float normalizePartVelocity2 = normalizePartVelocity(1, f5);
        float normalizePartVelocity3 = normalizePartVelocity(2, f4);
        float f6 = this.headWeight * normalizePartVelocity;
        float f7 = this.handWeight * normalizePartVelocity2;
        float f8 = this.hipWeight * normalizePartVelocity3;
        if (f6 >= f8 && f6 >= f7) {
            i = 0;
            f = f6;
        } else if (f8 >= f7) {
            f = f8;
            i = 2;
        } else {
            i = 1;
            f = f7;
        }
        MoveResult moveResult = new MoveResult(i, f, f3, f5, f4);
        moveResult.headIntensity = normalizePartVelocity;
        moveResult.handIntensity = normalizePartVelocity2;
        moveResult.hipIntensity = normalizePartVelocity3;
        return moveResult;
    }

    private double cappedDistance(float[] fArr, float[] fArr2, int i, double d) {
        double d2 = fArr[i] - fArr2[i];
        int i2 = i + 1;
        double d3 = fArr[i2] - fArr2[i2];
        return Math.min(Math.sqrt((d2 * d2) + (d3 * d3)) / d, this.maxFrameMove);
    }

    private double cappedPointDistance(float[] fArr, float[] fArr2, double d) {
        double d2 = fArr[0] - fArr2[0];
        double d3 = fArr[1] - fArr2[1];
        return Math.min(Math.sqrt((d2 * d2) + (d3 * d3)) / d, this.maxFrameMove);
    }

    private float dist2(float f, float f2, float f3, float f4) {
        float f5 = f - f3;
        float f6 = f2 - f4;
        return (float) Math.sqrt((f5 * f5) + (f6 * f6));
    }

    private double distanceRelative(float[] fArr, float[] fArr2, int i, float[] fArr3, float[] fArr4, double d) {
        int i2 = i * 2;
        double d2 = (fArr[i2] - fArr3[0]) - (fArr2[i2] - fArr4[0]);
        int i3 = i2 + 1;
        double d3 = (fArr[i3] - fArr3[1]) - (fArr2[i3] - fArr4[1]);
        return Math.min(Math.sqrt((d2 * d2) + (d3 * d3)) / d, this.maxFrameMove);
    }

    private void fill(float[] fArr, float[] fArr2, int i, NormalizedLandmark normalizedLandmark) {
        int i2 = i * 2;
        fArr[i2] = normalizedLandmark.x();
        fArr[i2 + 1] = normalizedLandmark.y();
        fArr2[i] = normalizedLandmark.visibility().orElse(Float.valueOf(0.0f)).floatValue();
    }

    private boolean hasAnyReliable(float[] fArr) {
        float f = fArr[0];
        float f2 = this.minVisibility;
        return f >= f2 || fArr[1] >= f2 || fArr[2] >= f2 || fArr[3] >= f2 || fArr[4] >= f2;
    }

    private float[] hipReference(float[] fArr, float[] fArr2) {
        float f = fArr2[3];
        float f2 = this.minVisibility;
        boolean z = f >= f2;
        boolean z2 = fArr2[4] >= f2;
        if (z && z2) {
            return new float[]{(fArr[6] + fArr[8]) / 2.0f, (fArr[7] + fArr[9]) / 2.0f, 0.0f};
        }
        if (z) {
            return new float[]{fArr[6], fArr[7], 1.0f};
        }
        if (z2) {
            return new float[]{fArr[8], fArr[9], 2.0f};
        }
        return null;
    }

    private float median(float[] fArr) {
        float[] fArr2 = (float[]) fArr.clone();
        Arrays.sort(fArr2);
        return fArr2[fArr2.length / 2];
    }

    private float normalizePartVelocity(int i, float f) {
        if (f < this.deadZone) {
            return 0.0f;
        }
        Deque<Float> deque = this.partVelocityRings.get(i);
        deque.addLast(Float.valueOf(f));
        while (deque.size() > PART_RING_SIZE) {
            deque.removeFirst();
        }
        float percentile = percentile(deque, 0.9f);
        float f2 = PART_REF_MAX;
        float max = Math.max(0.2f, Math.min(PART_REF_MAX, percentile));
        float[] fArr = this.partRefVelocity;
        float f3 = (fArr[i] * 0.9f) + (max * 0.1f);
        float f4 = f3 >= 0.2f ? f3 : 0.2f;
        if (f4 <= PART_REF_MAX) {
            f2 = f4;
        }
        fArr[i] = f2;
        return Math.min(f / f2, PART_INTENSITY_MAX);
    }

    private float percentile(Deque<Float> deque, float f) {
        if (deque.isEmpty()) {
            return 1.0f;
        }
        ArrayList arrayList = new ArrayList(deque);
        Collections.sort(arrayList);
        return ((Float) arrayList.get(Math.max(0, Math.min(arrayList.size() - 1, Math.round(f * (arrayList.size() - 1)))))).floatValue();
    }

    private int updateMissing(long j) {
        long j2 = this.lastSeenTimestamp;
        if (j2 >= 0 && j > j2) {
            float pow = (float) (this.smoothedSpeed * Math.pow(this.missingDecay, (j - j2) / 100.0d));
            this.smoothedSpeed = pow;
            if (pow < 1.0f) {
                this.smoothedSpeed = 0.0f;
            }
        }
        this.lastSeenTimestamp = j;
        return Math.round(this.smoothedSpeed);
    }

    public void clear() {
        reset();
    }

    public int getSpeed() {
        return Math.round(this.smoothedSpeed);
    }

    public void reset() {
        this.history.clear();
        this.lastTimestamp = -1L;
        this.lastSeenTimestamp = -1L;
        this.hasLastFrame = false;
        this.smoothedSpeed = 0.0f;
        this.scaleSmooth = -1.0f;
        Arrays.fill(this.handArmLengthRef, -1.0f);
        int[] iArr = this.handArmLenInitCount;
        iArr[0] = 0;
        iArr[1] = 0;
    }

    public void resetCalibration() {
        Iterator<Deque<Float>> it = this.partVelocityRings.iterator();
        while (it.hasNext()) {
            it.next().clear();
        }
        Arrays.fill(this.partRefVelocity, PART_REF_DEFAULT);
    }

    public void setDeadZone(float f) {
        this.deadZone = f;
    }

    public void setEnabledParts(boolean z, boolean z2, boolean z3) {
        this.enableHead = z;
        this.enableHand = z2;
        this.enableHip = z3;
    }

    public void setPriorityScale(float f) {
        this.priorityScale = Math.max(PART_REF_MAX, f);
    }

    public void setSmooth(float f, float f2) {
        this.attackAlpha = Math.max(0.05f, Math.min(1.0f, f));
        this.releaseAlpha = Math.max(0.02f, Math.min(1.0f, f2));
    }

    public void setWeight(float f, float f2, float f3) {
        this.headWeight = f;
        this.handWeight = f2;
        this.hipWeight = f3;
    }

    public int update(List<NormalizedLandmark> list, long j) {
        float f;
        HistoryFrame peekLast;
        float f2;
        HistoryFrame peekLast2;
        if (list == null || list.size() < 33) {
            return updateMissing(j);
        }
        float[] fArr = new float[14];
        float[] fArr2 = new float[7];
        fill(fArr, fArr2, 0, list.get(0));
        fill(fArr, fArr2, 1, list.get(15));
        fill(fArr, fArr2, 2, list.get(16));
        fill(fArr, fArr2, 3, list.get(23));
        fill(fArr, fArr2, 4, list.get(24));
        fill(fArr, fArr2, 5, list.get(11));
        fill(fArr, fArr2, 6, list.get(12));
        if (!hasAnyReliable(fArr2)) {
            return updateMissing(j);
        }
        boolean z = this.hasLastFrame;
        if (z && j == this.lastTimestamp) {
            return Math.round(this.smoothedSpeed);
        }
        if (z && (peekLast2 = this.history.peekLast()) != null && Arrays.equals(peekLast2.pts, fArr)) {
            return Math.round(this.smoothedSpeed);
        }
        boolean z2 = this.hasLastFrame;
        if (z2 && j - this.lastTimestamp > this.maxFrameGapMs) {
            this.history.clear();
            this.hasLastFrame = false;
            z2 = false;
        }
        if (z2 && j - this.lastTimestamp > 5000) {
            resetCalibration();
        }
        float calculateBodyScale = calculateBodyScale(fArr, fArr2);
        if (!this.hasLastFrame || (peekLast = this.history.peekLast()) == null) {
            f = calculateBodyScale;
        } else {
            float max = Math.max(peekLast.bodyScale, calculateBodyScale);
            if (max < 0.01f) {
                max = 0.01f;
            }
            int[] iArr = {0, 5, 6, 3, 4};
            int i = 0;
            int i2 = 0;
            for (int i3 = 5; i < i3; i3 = 5) {
                int i4 = iArr[i];
                if (fArr2[i4] < this.minVisibility || peekLast.visibility[i4] < this.minVisibility) {
                    f2 = calculateBodyScale;
                } else {
                    int i5 = i4 * 2;
                    int i6 = i5 + 1;
                    f2 = calculateBodyScale;
                    if (dist2(fArr[i5], fArr[i6], peekLast.pts[i5], peekLast.pts[i6]) / max > 0.8f) {
                        i2++;
                    }
                }
                i++;
                calculateBodyScale = f2;
            }
            f = calculateBodyScale;
            if (i2 >= 2) {
                LogUtil.d("PoseMoveSpeedTracker2 " + String.format(Locale.US, "镜头切换：%d 个身体锚点大位移，重建窗口，本帧为新基准", Integer.valueOf(i2)));
                this.history.clear();
                this.hasLastFrame = false;
                Arrays.fill(this.handArmLengthRef, -1.0f);
            }
        }
        HistoryFrame historyFrame = new HistoryFrame(j, fArr, fArr2, f);
        LogUtil.w("PoseMoveSpeedTracker2 frame:" + historyFrame);
        this.history.addLast(historyFrame);
        while (this.history.size() > this.maxHistorySize) {
            this.history.removeFirst();
        }
        if (!this.hasLastFrame) {
            this.hasLastFrame = true;
            this.lastTimestamp = j;
            this.lastSeenTimestamp = j;
            return Math.round(this.smoothedSpeed);
        }
        MoveResult calculateWindowVelocity = calculateWindowVelocity();
        this.lastTimestamp = j;
        this.lastSeenTimestamp = j;
        float min = Math.min(100.0f, (calculateWindowVelocity.move / Math.max(Math.max(this.headWeight, Math.max(this.handWeight, this.hipWeight)), 0.1f)) * this.priorityScale);
        float f3 = this.smoothedSpeed;
        float f4 = f3 + ((min - f3) * (min >= f3 ? this.attackAlpha : this.releaseAlpha));
        this.smoothedSpeed = f4;
        if (min == 0.0f && f4 < 1.0f) {
            this.smoothedSpeed = 0.0f;
            f4 = 0.0f;
        }
        int round = Math.round(Math.max(0.0f, Math.min(100.0f, f4)));
        LogUtil.d("PoseMoveSpeedTracker2 " + String.format(Locale.US, "ts=%d scale=%.3f noseV=%.3f handV=%.3f hipV=%.3f nI=%.2f hI=%.2f pI=%.2f v=%.2f target=%.1f out=%d", Long.valueOf(j), Float.valueOf(f), Float.valueOf(calculateWindowVelocity.headMove), Float.valueOf(calculateWindowVelocity.handMove), Float.valueOf(calculateWindowVelocity.hipMove), Float.valueOf(calculateWindowVelocity.headIntensity), Float.valueOf(calculateWindowVelocity.handIntensity), Float.valueOf(calculateWindowVelocity.hipIntensity), Float.valueOf(calculateWindowVelocity.move), Float.valueOf(min), Integer.valueOf(round)));
        return round;
    }
}
