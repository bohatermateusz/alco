package com.hismith.hismithapp.pose;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import androidx.core.content.ContextCompat;
import androidx.core.internal.view.SupportMenu;
import androidx.core.view.InputDeviceCompat;
import com.google.mediapipe.tasks.components.containers.Connection;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import com.hismith.hismithapp.R;
import java.util.List;
import java.util.Objects;

/* loaded from: classes3.dex */
public class OverlayView extends View {
    private static final float LANDMARK_STROKE_WIDTH = 8.0f;
    private static final int MAX_POSES = 5;
    private static final int[] POSE_COLORS = {InputDeviceCompat.SOURCE_ANY, -16711681, -65281, Color.rgb(0, 255, 0), Color.rgb(255, 165, 0)};
    private int activePart;
    private final Paint activePartPaint;
    private int imageHeight;
    private int imageWidth;
    private final Paint linePaint;
    private int mainPersonIndex;
    private final Paint pointPaint;
    private PoseLandmarkerResult results;
    private float scaleFactor;

    public OverlayView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mainPersonIndex = -1;
        this.activePart = -1;
        this.pointPaint = new Paint();
        this.linePaint = new Paint();
        this.activePartPaint = new Paint();
        this.scaleFactor = 1.0f;
        this.imageWidth = 1;
        this.imageHeight = 1;
        initPaints();
    }

    private void drawActiveDot(Canvas canvas, float f, float f2, float f3) {
        float f4 = this.scaleFactor;
        canvas.drawCircle(f * this.imageWidth * f4, f2 * this.imageHeight * f4, f3, this.activePartPaint);
    }

    private void initPaints() {
        this.linePaint.setColor(ContextCompat.getColor((Context) Objects.requireNonNull(getContext()), R.color.app_style_color));
        this.linePaint.setStrokeWidth(8.0f);
        this.linePaint.setStyle(Paint.Style.STROKE);
        this.pointPaint.setColor(InputDeviceCompat.SOURCE_ANY);
        this.pointPaint.setStrokeWidth(8.0f);
        this.pointPaint.setStyle(Paint.Style.FILL);
        this.activePartPaint.setColor(SupportMenu.CATEGORY_MASK);
        this.activePartPaint.setStyle(Paint.Style.FILL);
    }

    public void clear() {
        this.results = null;
        this.mainPersonIndex = -1;
        this.activePart = -1;
        this.pointPaint.reset();
        this.linePaint.reset();
        this.activePartPaint.reset();
        invalidate();
        initPaints();
    }

    public void clearPoseOverlay() {
        this.results = null;
        invalidate();
    }

    @Override // android.view.View
    public void draw(Canvas canvas) {
        int i;
        List<NormalizedLandmark> list;
        super.draw(canvas);
        PoseLandmarkerResult poseLandmarkerResult = this.results;
        if (poseLandmarkerResult != null) {
            PoseResultFormatter.format(poseLandmarkerResult);
            List<List<NormalizedLandmark>> landmarks = this.results.landmarks();
            int min = Math.min(landmarks.size(), 5);
            for (int i2 = 0; i2 < min; i2++) {
                List<NormalizedLandmark> list2 = landmarks.get(i2);
                if (list2 != null && list2.size() >= 33) {
                    int[] iArr = POSE_COLORS;
                    int i3 = iArr[i2 % iArr.length];
                    int i4 = this.mainPersonIndex;
                    Paint paint = this.pointPaint;
                    if (i2 == i4) {
                        paint.setColor(-1);
                        this.linePaint.setColor(-1);
                        this.linePaint.setStrokeWidth(12.0f);
                    } else {
                        paint.setColor(i3);
                        this.linePaint.setColor(i3);
                        this.linePaint.setStrokeWidth(8.0f);
                    }
                    for (NormalizedLandmark normalizedLandmark : list2) {
                        canvas.drawPoint(normalizedLandmark.x() * this.imageWidth * this.scaleFactor, normalizedLandmark.y() * this.imageHeight * this.scaleFactor, this.pointPaint);
                    }
                    for (Connection connection : PoseLandmarker.POSE_LANDMARKS) {
                        int start = connection.start();
                        int end = connection.end();
                        NormalizedLandmark normalizedLandmark2 = list2.get(start);
                        NormalizedLandmark normalizedLandmark3 = list2.get(end);
                        canvas.drawLine(normalizedLandmark2.x() * this.imageWidth * this.scaleFactor, normalizedLandmark2.y() * this.imageHeight * this.scaleFactor, normalizedLandmark3.x() * this.imageWidth * this.scaleFactor, normalizedLandmark3.y() * this.imageHeight * this.scaleFactor, this.linePaint);
                    }
                }
            }
            if (this.activePart < 0 || (i = this.mainPersonIndex) < 0 || i >= landmarks.size() || (list = landmarks.get(this.mainPersonIndex)) == null || list.size() < 33) {
                return;
            }
            int i5 = this.activePart;
            if (i5 == 0) {
                drawActiveDot(canvas, list.get(0).x(), list.get(0).y(), 12.0f);
                return;
            }
            if (i5 == 1) {
                drawActiveDot(canvas, list.get(15).x(), list.get(15).y(), 12.0f);
            } else if (i5 == 2) {
                drawActiveDot(canvas, list.get(16).x(), list.get(16).y(), 12.0f);
            } else if (i5 == 3) {
                drawActiveDot(canvas, (list.get(23).x() + list.get(24).x()) / 2.0f, (list.get(23).y() + list.get(24).y()) / 2.0f, 12.0f);
            }
        }
    }

    public void setActivePart(int i) {
        this.activePart = i;
        invalidate();
    }

    public void setResults(PoseLandmarkerResult poseLandmarkerResult, int i, int i2) {
        setResults(poseLandmarkerResult, i, i2, RunningMode.IMAGE);
    }

    public void setResults(PoseLandmarkerResult poseLandmarkerResult, int i, int i2, RunningMode runningMode) {
        setResults(poseLandmarkerResult, i, i2, runningMode, -1);
    }

    public void setResults(PoseLandmarkerResult poseLandmarkerResult, int i, int i2, RunningMode runningMode, int i3) {
        this.results = poseLandmarkerResult;
        this.mainPersonIndex = i3;
        this.imageHeight = i;
        this.imageWidth = i2;
        int width = getWidth();
        int height = getHeight();
        if (width == 0 || height == 0) {
            this.scaleFactor = 1.0f;
        } else if (runningMode == RunningMode.IMAGE || runningMode == RunningMode.VIDEO) {
            this.scaleFactor = Math.min((width * 1.0f) / i2, (height * 1.0f) / i);
        } else {
            this.scaleFactor = Math.max((width * 1.0f) / i2, (height * 1.0f) / i);
        }
        invalidate();
    }
}
