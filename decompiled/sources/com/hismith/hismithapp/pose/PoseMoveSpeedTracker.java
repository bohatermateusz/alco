package com.hismith.hismithapp.pose;

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Locale;

/* loaded from: classes3.dex */
public class PoseMoveSpeedTracker {
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
    public static final float MAIN_PERSON_LOCK_DISTANCE = 0.2f;
    private static final float MAX_HAND_MOVE_PER_FRAME = 1.1f;
    private static final float MIN_HAND_VISIBILITY = 0.65f;
    private static final int NOSE = 0;
    private static final int POINT_COUNT = 7;
    private static final int P_LEFT_HIP = 3;
    private static final int P_LEFT_SHOULDER = 5;
    private static final int P_LEFT_WRIST = 1;
    private static final int P_NOSE = 0;
    private static final int P_RIGHT_HIP = 4;
    private static final int P_RIGHT_SHOULDER = 6;
    private static final int P_RIGHT_WRIST = 2;
    private static final float REF_MAX = 20.0f;
    private static final float REF_MIN = 0.8f;
    private static final float REF_PERCENTILE = 0.85f;
    private static final int RIGHT_HIP = 24;
    private static final int RIGHT_SHOULDER = 12;
    private static final int RIGHT_WRIST = 16;
    private static final int RING_SIZE = 40;
    private static final float SHOT_CUT_THRESHOLD = 0.8f;
    private static final String TAG = "PoseMoveSpeedTracker";
    private float minVisibility = 0.5f;
    private int maxHistorySize = 6;
    private long maxFrameGapMs = 300;
    private float deadZone = 0.15f;
    private float maxFrameMove = 1.5f;
    private float attackAlpha = 0.55f;
    private float releaseAlpha = 0.25f;
    private float missingDecay = 0.8f;
    private float headWeight = 1.6f;
    private float hipWeight = 2.5f;
    private float handWeight = 1.2f;
    private boolean enableHead = true;
    private boolean enableHand = true;
    private boolean enableHip = true;
    private final float[] handArmLengthRef = {-1.0f, -1.0f};
    private final float[][] handArmLenInit = {new float[5], new float[5]};
    private final int[] handArmLenInitCount = {0, 0};
    private boolean adaptive = false;
    private float fixedMaxSpeed = 8.0f;
    private float refVelocity = 5.0f;
    private final Deque<Float> velocityRing = new ArrayDeque();
    private final Deque<HistoryFrame> history = new ArrayDeque();
    private long lastTimestamp = -1;
    private long lastSeenTimestamp = -1;
    private boolean hasLastFrame = false;
    private float smoothedSpeed = 0.0f;
    private float scaleSmooth = -1.0f;
    private int currentPart = -1;

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
        public float handMove;
        public float headMove;
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
        ArrayList arrayList;
        double d2;
        float[] fArr;
        float[] fArr2;
        double d3;
        int i2;
        double d4;
        float[] fArr3;
        float[] fArr4;
        float[] fArr5;
        PoseMoveSpeedTracker poseMoveSpeedTracker = this;
        ArrayList arrayList2 = new ArrayList(poseMoveSpeedTracker.history);
        int i3 = 2;
        if (arrayList2.size() < 2) {
            return new MoveResult(0, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        double d5 = 0.0d;
        double d6 = 0.0d;
        int i4 = 1;
        double d7 = 0.0d;
        double d8 = 0.0d;
        double d9 = 0.0d;
        double d10 = 0.0d;
        while (i4 < arrayList2.size()) {
            HistoryFrame historyFrame3 = (HistoryFrame) arrayList2.get(i4 - 1);
            HistoryFrame historyFrame4 = (HistoryFrame) arrayList2.get(i4);
            int i5 = i3;
            long j = historyFrame4.timestamp - historyFrame3.timestamp;
            if (j <= 0) {
                arrayList = arrayList2;
            } else {
                double d11 = (j / 1000.0d) + d5;
                double max = Math.max(historyFrame3.bodyScale, historyFrame4.bodyScale);
                if (max < 0.01d) {
                    max = 0.01d;
                }
                if (!poseMoveSpeedTracker.enableHead || historyFrame4.visibility[0] < poseMoveSpeedTracker.minVisibility || historyFrame3.visibility[0] < poseMoveSpeedTracker.minVisibility) {
                    historyFrame = historyFrame3;
                    historyFrame2 = historyFrame4;
                    d = max;
                } else {
                    historyFrame = historyFrame3;
                    historyFrame2 = historyFrame4;
                    d = max;
                    d6 += poseMoveSpeedTracker.cappedDistance(historyFrame4.pts, historyFrame3.pts, 0, d);
                }
                double d12 = d6;
                float[] hipReference = poseMoveSpeedTracker.hipReference(historyFrame2.pts, historyFrame2.visibility);
                float[] hipReference2 = poseMoveSpeedTracker.hipReference(historyFrame.pts, historyFrame.visibility);
                if (hipReference != null && hipReference2 != null && hipReference[i5] == hipReference2[i5] && poseMoveSpeedTracker.enableHip) {
                    d7 += poseMoveSpeedTracker.cappedPointDistance(hipReference, hipReference2, d);
                }
                if (poseMoveSpeedTracker.enableHand) {
                    int i6 = i5;
                    int i7 = 1;
                    int[] iArr = {1, i6};
                    int i8 = 0;
                    double d13 = -1.0d;
                    while (i8 < i6) {
                        int i9 = iArr[i8];
                        int i10 = i9 == i7 ? 0 : i7;
                        ArrayList arrayList3 = arrayList2;
                        int i11 = i9 == i7 ? 5 : 6;
                        if (historyFrame2.visibility[i9] >= MIN_HAND_VISIBILITY && historyFrame.visibility[i9] >= MIN_HAND_VISIBILITY) {
                            if (hipReference != null && hipReference2 != null && hipReference[2] == hipReference2[2]) {
                                fArr3 = hipReference;
                                fArr5 = fArr3;
                                fArr4 = hipReference2;
                            } else if (historyFrame2.visibility[i11] >= poseMoveSpeedTracker.minVisibility && historyFrame.visibility[i11] >= poseMoveSpeedTracker.minVisibility) {
                                int i12 = i11 * 2;
                                int i13 = i12 + 1;
                                fArr3 = new float[]{historyFrame2.pts[i12], historyFrame2.pts[i13]};
                                fArr4 = new float[]{historyFrame.pts[i12], historyFrame.pts[i13]};
                                fArr5 = hipReference;
                            }
                            fArr2 = hipReference2;
                            i2 = i8;
                            float[] fArr6 = fArr4;
                            poseMoveSpeedTracker = this;
                            double d14 = d;
                            double distanceRelative = poseMoveSpeedTracker.distanceRelative(historyFrame2.pts, historyFrame.pts, i9, fArr3, fArr6, d14);
                            if (distanceRelative > 1.100000023841858d) {
                                d3 = d14;
                                fArr = fArr5;
                                d4 = d11;
                                d13 = d13;
                                i8 = i2 + 1;
                                arrayList2 = arrayList3;
                                hipReference = fArr;
                                hipReference2 = fArr2;
                                d11 = d4;
                                d = d3;
                                i7 = 1;
                                i6 = 2;
                            } else {
                                if (historyFrame2.visibility[i11] >= poseMoveSpeedTracker.minVisibility) {
                                    int i14 = i11 * 2;
                                    float f2 = historyFrame2.pts[i14];
                                    float f3 = historyFrame2.pts[i14 + 1];
                                    fArr = fArr5;
                                    int i15 = i9 * 2;
                                    d4 = d11;
                                    float dist2 = poseMoveSpeedTracker.dist2(f2, f3, historyFrame2.pts[i15], historyFrame2.pts[i15 + 1]) / ((float) d14);
                                    if (dist2 >= 0.5f && dist2 <= 2.0f) {
                                        float[] fArr7 = poseMoveSpeedTracker.handArmLengthRef;
                                        float f4 = fArr7[i10];
                                        if (f4 <= 0.0f) {
                                            int[] iArr2 = poseMoveSpeedTracker.handArmLenInitCount;
                                            int i16 = iArr2[i10];
                                            if (i16 < 5) {
                                                float[] fArr8 = poseMoveSpeedTracker.handArmLenInit[i10];
                                                d3 = d14;
                                                int i17 = i16 + 1;
                                                iArr2[i10] = i17;
                                                fArr8[i16] = dist2;
                                                if (i17 == 5) {
                                                    fArr7[i10] = Math.max(poseMoveSpeedTracker.median(fArr8), 0.8f);
                                                }
                                            }
                                        } else if (dist2 >= ARM_LEN_MIN_RATIO * f4 && dist2 <= ARM_LEN_MAX_RATIO * f4) {
                                            fArr7[i10] = Math.max((f4 * 0.8f) + (dist2 * 0.2f), 0.8f);
                                        }
                                        d3 = d14;
                                    }
                                    d3 = d14;
                                    d13 = d13;
                                    i8 = i2 + 1;
                                    arrayList2 = arrayList3;
                                    hipReference = fArr;
                                    hipReference2 = fArr2;
                                    d11 = d4;
                                    d = d3;
                                    i7 = 1;
                                    i6 = 2;
                                } else {
                                    d3 = d14;
                                    fArr = fArr5;
                                    d4 = d11;
                                }
                                double d15 = d13;
                                d13 = distanceRelative > d15 ? distanceRelative : d15;
                                float f5 = poseMoveSpeedTracker.maxFrameMove;
                                if (i9 == 1) {
                                    d10 += Math.min(distanceRelative, f5);
                                } else {
                                    d9 += Math.min(distanceRelative, f5);
                                }
                                i8 = i2 + 1;
                                arrayList2 = arrayList3;
                                hipReference = fArr;
                                hipReference2 = fArr2;
                                d11 = d4;
                                d = d3;
                                i7 = 1;
                                i6 = 2;
                            }
                        }
                        fArr = hipReference;
                        fArr2 = hipReference2;
                        d3 = d;
                        i2 = i8;
                        d4 = d11;
                        d13 = d13;
                        i8 = i2 + 1;
                        arrayList2 = arrayList3;
                        hipReference = fArr;
                        hipReference2 = fArr2;
                        d11 = d4;
                        d = d3;
                        i7 = 1;
                        i6 = 2;
                    }
                    arrayList = arrayList2;
                    d2 = d11;
                    double d16 = d13;
                    if (d16 >= 0.0d && arrayList.size() >= 3) {
                        d8 += Math.min(d16, poseMoveSpeedTracker.maxFrameMove);
                    }
                } else {
                    arrayList = arrayList2;
                    d2 = d11;
                }
                d6 = d12;
                d5 = d2;
            }
            i4++;
            arrayList2 = arrayList;
            i3 = 2;
        }
        if (d5 <= 0.0d) {
            return new MoveResult(0, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        float f6 = (float) (d6 / d5);
        float f7 = (float) (d7 / d5);
        float f8 = (float) (d8 / d5);
        char c = (d9 <= d10 || d9 <= 0.0d) ? d10 > 0.0d ? (char) 0 : (char) 65535 : (char) 1;
        float f9 = poseMoveSpeedTracker.headWeight * f6;
        float f10 = poseMoveSpeedTracker.hipWeight * f7;
        float f11 = poseMoveSpeedTracker.handWeight * f8;
        if (f9 >= f10 && f9 >= f11) {
            f = f9;
            i = 0;
        } else if (f10 >= f11) {
            i = 3;
            f = f10;
        } else {
            i = c == 1 ? 2 : 1;
            f = f11;
        }
        return new MoveResult(i, f, f6, f8, f7);
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

    private static float dist2d(float f, float f2, float f3, float f4) {
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

    private float mapVelocity(float f) {
        Deque<Float> deque;
        if (!this.adaptive) {
            return Math.max(0.0f, Math.min(100.0f, (f / Math.max(this.fixedMaxSpeed, 0.1f)) * 100.0f));
        }
        this.velocityRing.addLast(Float.valueOf(f));
        while (true) {
            int size = this.velocityRing.size();
            deque = this.velocityRing;
            if (size <= 40) {
                break;
            }
            deque.removeFirst();
        }
        float percentile = percentile(deque, REF_PERCENTILE);
        float f2 = REF_MAX;
        float f3 = 0.8f;
        float max = (this.refVelocity * 0.8f) + (Math.max(0.8f, Math.min(REF_MAX, percentile)) * 0.2f);
        this.refVelocity = max;
        if (max < 0.8f) {
            this.refVelocity = 0.8f;
        } else {
            f3 = max;
        }
        if (f3 > REF_MAX) {
            this.refVelocity = REF_MAX;
        } else {
            f2 = f3;
        }
        return Math.max(0.0f, Math.min(100.0f, (f / f2) * 100.0f));
    }

    private float median(float[] fArr) {
        float[] fArr2 = (float[]) fArr.clone();
        Arrays.sort(fArr2);
        return fArr2[fArr2.length / 2];
    }

    private float percentile(Deque<Float> deque, float f) {
        if (deque.isEmpty()) {
            return this.fixedMaxSpeed;
        }
        ArrayList arrayList = new ArrayList(deque);
        Collections.sort(arrayList);
        return ((Float) arrayList.get(Math.max(0, Math.min(arrayList.size() - 1, Math.round(f * (arrayList.size() - 1)))))).floatValue();
    }

    public static float[] personFingerprint(List<NormalizedLandmark> list) {
        NormalizedLandmark normalizedLandmark;
        if (list == null || list.size() <= 0 || (normalizedLandmark = list.get(0)) == null) {
            return null;
        }
        return new float[]{normalizedLandmark.x(), normalizedLandmark.y()};
    }

    private static float poseScore(List<NormalizedLandmark> list) {
        NormalizedLandmark normalizedLandmark;
        int[] iArr = {0, 11, 12, 23, 24, 15, 16};
        float f = 0.0f;
        Float valueOf = Float.valueOf(0.0f);
        float f2 = 0.0f;
        float f3 = 0.0f;
        float f4 = 0.0f;
        int i = 0;
        int i2 = 0;
        for (int i3 = 0; i3 < 7; i3++) {
            int i4 = iArr[i3];
            if (i4 < list.size() && (normalizedLandmark = list.get(i4)) != null) {
                f2 += normalizedLandmark.visibility().orElse(valueOf).floatValue();
                i++;
                f3 += normalizedLandmark.x();
                f4 += normalizedLandmark.y();
                i2++;
            }
        }
        if (i == 0 || i2 == 0) {
            return -1.0f;
        }
        float f5 = f2 / i;
        float f6 = i2;
        float f7 = (f3 / f6) - 0.5f;
        float f8 = (f4 / f6) - 0.5f;
        float sqrt = (float) Math.sqrt((f7 * f7) + (f8 * f8));
        float f9 = torsoLength(list);
        if (f9 > 0.01f) {
            int[] iArr2 = {15, 16};
            int i5 = 0;
            for (int i6 = 0; i6 < 2; i6++) {
                int i7 = iArr2[i6];
                int i8 = i7 == 15 ? 11 : 12;
                if (i7 < list.size() && i8 < list.size()) {
                    NormalizedLandmark normalizedLandmark2 = list.get(i7);
                    NormalizedLandmark normalizedLandmark3 = list.get(i8);
                    if (normalizedLandmark2 != null && normalizedLandmark3 != null && normalizedLandmark2.visibility().orElse(valueOf).floatValue() >= 0.5f && normalizedLandmark3.visibility().orElse(valueOf).floatValue() >= 0.5f) {
                        float dist2d = dist2d(normalizedLandmark2.x(), normalizedLandmark2.y(), normalizedLandmark3.x(), normalizedLandmark3.y()) / f9;
                        if (dist2d < 0.25f || dist2d > 2.2f) {
                            i5++;
                        }
                    }
                }
            }
            f = i5 * 0.25f;
        }
        return (f5 - (sqrt * 0.5f)) - f;
    }

    public static int selectMainPersonIndex(List<List<NormalizedLandmark>> list) {
        int i = -1;
        if (list != null && !list.isEmpty()) {
            float f = -1.0f;
            for (int i2 = 0; i2 < list.size(); i2++) {
                List<NormalizedLandmark> list2 = list.get(i2);
                if (list2 != null && !list2.isEmpty()) {
                    float poseScore = poseScore(list2);
                    if (poseScore > f) {
                        i = i2;
                        f = poseScore;
                    }
                }
            }
        }
        return i;
    }

    private static float torsoLength(List<NormalizedLandmark> list) {
        if (list != null && list.size() > 24) {
            NormalizedLandmark normalizedLandmark = list.get(11);
            NormalizedLandmark normalizedLandmark2 = list.get(12);
            NormalizedLandmark normalizedLandmark3 = list.get(23);
            NormalizedLandmark normalizedLandmark4 = list.get(24);
            if (normalizedLandmark != null && normalizedLandmark2 != null && normalizedLandmark3 != null && normalizedLandmark4 != null) {
                return dist2d((normalizedLandmark.x() + normalizedLandmark2.x()) / 2.0f, (normalizedLandmark.y() + normalizedLandmark2.y()) / 2.0f, (normalizedLandmark3.x() + normalizedLandmark4.x()) / 2.0f, (normalizedLandmark3.y() + normalizedLandmark4.y()) / 2.0f);
            }
        }
        return -1.0f;
    }

    private int updateMissing(long j) {
        this.currentPart = -1;
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

    public int getCurrentPart() {
        return this.currentPart;
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
        this.currentPart = -1;
        this.velocityRing.clear();
        this.refVelocity = this.fixedMaxSpeed;
        Arrays.fill(this.handArmLengthRef, -1.0f);
        int[] iArr = this.handArmLenInitCount;
        iArr[0] = 0;
        iArr[1] = 0;
    }

    public void setAdaptive(boolean z) {
        this.adaptive = z;
        if (z) {
            return;
        }
        this.velocityRing.clear();
        this.refVelocity = this.fixedMaxSpeed;
    }

    public void setDeadZone(float f) {
        this.deadZone = f;
    }

    public void setEnabledParts(boolean z, boolean z2, boolean z3) {
        this.enableHead = z;
        this.enableHand = z2;
        this.enableHip = z3;
    }

    public void setMaxSpeedFor100(float f) {
        float max = Math.max(0.1f, f);
        this.fixedMaxSpeed = max;
        if (this.adaptive) {
            return;
        }
        this.refVelocity = max;
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
        HistoryFrame peekLast;
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
        if (z && j - this.lastTimestamp > this.maxFrameGapMs) {
            this.history.clear();
            this.hasLastFrame = false;
        }
        float calculateBodyScale = calculateBodyScale(fArr, fArr2);
        if (this.hasLastFrame && (peekLast = this.history.peekLast()) != null) {
            float max = Math.max(peekLast.bodyScale, calculateBodyScale);
            if (max < 0.01f) {
                max = 0.01f;
            }
            int[] iArr = {0, 5, 6, 3, 4};
            int i = 0;
            int i2 = 0;
            for (int i3 = 5; i < i3; i3 = 5) {
                int i4 = iArr[i];
                if (fArr2[i4] >= this.minVisibility && peekLast.visibility[i4] >= this.minVisibility) {
                    int i5 = i4 * 2;
                    int i6 = i5 + 1;
                    if (dist2(fArr[i5], fArr[i6], peekLast.pts[i5], peekLast.pts[i6]) / max > 0.8f) {
                        i2++;
                    }
                }
                i++;
            }
            if (i2 >= 2) {
                LogUtil.d("PoseMoveSpeedTracker " + String.format(Locale.US, "镜头切换：%d 个身体锚点大幅位移，重建窗口，本帧作为新基准", Integer.valueOf(i2)));
                this.history.clear();
                this.hasLastFrame = false;
                Arrays.fill(this.handArmLengthRef, -1.0f);
            }
        }
        HistoryFrame historyFrame = new HistoryFrame(j, fArr, fArr2, calculateBodyScale);
        LogUtil.w("PoseMoveSpeedTracker frame:" + historyFrame);
        this.history.addLast(historyFrame);
        while (this.history.size() > this.maxHistorySize) {
            this.history.removeFirst();
        }
        if (!this.hasLastFrame) {
            this.hasLastFrame = true;
            this.lastTimestamp = j;
            this.lastSeenTimestamp = j;
            this.currentPart = -1;
            return Math.round(this.smoothedSpeed);
        }
        MoveResult calculateWindowVelocity = calculateWindowVelocity();
        this.currentPart = calculateWindowVelocity.part;
        this.lastTimestamp = j;
        this.lastSeenTimestamp = j;
        float f = calculateWindowVelocity.move;
        if (f < this.deadZone) {
            f = 0.0f;
        }
        float mapVelocity = mapVelocity(f);
        float f2 = this.smoothedSpeed;
        float f3 = f2 + ((mapVelocity - f2) * (mapVelocity >= f2 ? this.attackAlpha : this.releaseAlpha));
        this.smoothedSpeed = f3;
        if (mapVelocity == 0.0f && f3 < 1.0f) {
            this.smoothedSpeed = 0.0f;
            f3 = 0.0f;
        }
        int round = Math.round(Math.max(0.0f, Math.min(100.0f, f3)));
        LogUtil.d("PoseMoveSpeedTracker " + String.format(Locale.US, "ts=%d scale=%.3f noseV=%.3f handV=%.3f hipV=%.3f v=%.3f ref=%.2f target=%.1f out=%d", Long.valueOf(j), Float.valueOf(calculateBodyScale), Float.valueOf(calculateWindowVelocity.headMove), Float.valueOf(calculateWindowVelocity.handMove), Float.valueOf(calculateWindowVelocity.hipMove), Float.valueOf(f), Float.valueOf(this.refVelocity), Float.valueOf(mapVelocity), Integer.valueOf(round)));
        return round;
    }
}
