package com.hismith.hismithapp.pose;

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import java.util.List;

/* loaded from: classes3.dex */
public class PoseResultFormatter {
    private static final String[] LANDMARK_NAMES = {"鼻子", "左眼内眼角", "左眼", "左眼外眼角", "右眼内眼角", "右眼", "右眼外眼角", "左耳", "右耳", "左嘴角", "右嘴角", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕", "左小指", "右小指", "左指尖", "右指尖", "左拇指", "右拇指", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "左脚跟", "右脚跟", "左脚尖", "右脚尖"};
    private int[] LANDMARK_POSITION = {0, 15, 16, 23, 24};

    public static String format(PoseLandmarkerResult poseLandmarkerResult) {
        if (poseLandmarkerResult == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder("PoseLandmarkerResult: ");
        List<List<NormalizedLandmark>> landmarks = poseLandmarkerResult.landmarks();
        int size = landmarks.size();
        sb.append(size).append(" person(s)\n");
        for (int i = 0; i < size; i++) {
            List<NormalizedLandmark> list = landmarks.get(i);
            sb.append("  Person #").append(i).append(":\n");
            int i2 = 0;
            while (i2 < list.size()) {
                NormalizedLandmark normalizedLandmark = list.get(i2);
                String[] strArr = LANDMARK_NAMES;
                sb.append(String.format("    [%02d] %s: x=%.4f y=%.4f z=%.4f vis=%.4f pre=%.4f\n", Integer.valueOf(i2), i2 < strArr.length ? strArr[i2] : "未知", Float.valueOf(normalizedLandmark.x()), Float.valueOf(normalizedLandmark.y()), Float.valueOf(normalizedLandmark.z()), normalizedLandmark.visibility().orElse(Float.valueOf(-1.0f)), normalizedLandmark.presence().orElse(Float.valueOf(-1.0f))));
                i2++;
            }
        }
        return sb.toString();
    }

    public static String formatShort(PoseLandmarkerResult poseLandmarkerResult) {
        if (poseLandmarkerResult == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder("Poses:");
        int size = poseLandmarkerResult.landmarks().size();
        sb.append(size).append("\n");
        int[] iArr = {0, 11, 12, 23, 24, 25, 26, 27, 28};
        for (int i = 0; i < size; i++) {
            sb.append("  Person ").append(i).append(": ");
            List<NormalizedLandmark> list = poseLandmarkerResult.landmarks().get(i);
            for (int i2 = 0; i2 < 9; i2++) {
                int i3 = iArr[i2];
                if (i3 < list.size()) {
                    NormalizedLandmark normalizedLandmark = list.get(i3);
                    sb.append(String.format("%s:(%.2f,%.2f) ", LANDMARK_NAMES[i3], Float.valueOf(normalizedLandmark.x()), Float.valueOf(normalizedLandmark.y())));
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
