package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayView_Five extends View {
    static final String TAG = "PlayView_Five";
    private float mCircleRadius;
    private LineElement mCurrentLine;
    private long mElapsed;
    private Handler mHandler;
    private int mHeight;
    private float mLaserX;
    private float mLaserY;
    private List<LineElement> mLines;
    final Paint mPaint;
    private float mStrokeWidth;
    private int mWidth;

    class LineElement {
        public static final int ALPHA_STEP = 8;
        public Paint mPaint;
        public Path mPath;
        float mPathWidth;
        public float mStartX = -1.0f;
        public float mStartY = -1.0f;
        public float mEndX = -1.0f;
        public float mEndY = -1.0f;
        public PointF[] mPoints = new PointF[4];

        public LineElement(float f) {
            Paint paint = new Paint();
            this.mPaint = paint;
            int i = 0;
            paint.setARGB(255, 255, 0, 0);
            this.mPaint.setAntiAlias(true);
            this.mPaint.setStrokeWidth(0.0f);
            this.mPaint.setStrokeCap(Paint.Cap.BUTT);
            this.mPaint.setStyle(Paint.Style.FILL);
            this.mPath = new Path();
            this.mPathWidth = f;
            while (true) {
                PointF[] pointFArr = this.mPoints;
                if (i >= pointFArr.length) {
                    return;
                }
                pointFArr[i] = new PointF();
                i++;
            }
        }

        private boolean caculatePoints(float f, float f2, float f3, float f4, float f5, PointF pointF, PointF pointF2) {
            double pow = Math.pow(f, 2.0d) + 1.0d;
            float f6 = f2 - f4;
            double d = ((f * 2.0f) * f6) - (2.0f * f3);
            double pow2 = Math.pow(d, 2.0d) - ((4.0d * pow) * ((Math.pow(f3, 2.0d) + Math.pow(f6, 2.0d)) - Math.pow(f5, 2.0d)));
            if (pow2 <= 0.0d) {
                return false;
            }
            double sqrt = Math.sqrt(pow2);
            double d2 = -d;
            double d3 = pow * 2.0d;
            pointF.x = (float) ((d2 + sqrt) / d3);
            pointF.y = (pointF.x * f) + f2;
            pointF2.x = (float) ((d2 - sqrt) / d3);
            pointF2.y = (f * pointF2.x) + f2;
            return true;
        }

        private void swapPoint(PointF pointF, PointF pointF2) {
            float f = pointF.x;
            pointF.x = pointF2.x;
            pointF2.x = f;
            float f2 = pointF.y;
            pointF.y = pointF2.y;
            pointF2.y = f2;
        }

        public void setAlpha(int i) {
            this.mPaint.setAlpha(i);
            this.mPathWidth = (i * this.mPathWidth) / 255.0f;
        }

        public void setPaint(Paint paint) {
            this.mPaint = paint;
        }

        public void updatePath() {
            this.mPath.reset();
            this.mPath.moveTo(this.mPoints[0].x, this.mPoints[0].y);
            this.mPath.lineTo(this.mPoints[1].x, this.mPoints[1].y);
            this.mPath.lineTo(this.mPoints[2].x, this.mPoints[2].y);
            this.mPath.lineTo(this.mPoints[3].x, this.mPoints[3].y);
            this.mPath.close();
        }

        public boolean updatePathPoints() {
            float f = this.mPathWidth / 2.0f;
            if (Math.abs(this.mEndX - this.mStartX) < 1.0f) {
                this.mPoints[0].x = this.mStartX + f;
                this.mPoints[0].y = this.mStartY - f;
                this.mPoints[1].x = this.mStartX - f;
                PointF[] pointFArr = this.mPoints;
                pointFArr[1].y = pointFArr[0].y;
                PointF[] pointFArr2 = this.mPoints;
                pointFArr2[2].x = pointFArr2[1].x;
                this.mPoints[2].y = this.mEndY + f;
                PointF[] pointFArr3 = this.mPoints;
                pointFArr3[3].x = pointFArr3[0].x;
                PointF[] pointFArr4 = this.mPoints;
                pointFArr4[3].y = pointFArr4[2].y;
            } else if (Math.abs(this.mEndY - this.mStartY) < 1.0f) {
                this.mPoints[0].x = this.mStartX - f;
                this.mPoints[0].y = this.mStartY - f;
                PointF[] pointFArr5 = this.mPoints;
                pointFArr5[1].x = pointFArr5[0].x;
                this.mPoints[1].y = this.mStartY + f;
                this.mPoints[2].x = this.mEndX + f;
                PointF[] pointFArr6 = this.mPoints;
                pointFArr6[2].y = pointFArr6[1].y;
                PointF[] pointFArr7 = this.mPoints;
                pointFArr7[3].x = pointFArr7[2].x;
                PointF[] pointFArr8 = this.mPoints;
                pointFArr8[3].y = pointFArr8[0].y;
            } else {
                float f2 = this.mEndY;
                float f3 = this.mStartY;
                float f4 = this.mEndX;
                float f5 = this.mStartX;
                float f6 = (f2 - f3) / (f4 - f5);
                float f7 = (-1.0f) / f6;
                float f8 = f3 - (f7 * f5);
                PointF[] pointFArr9 = this.mPoints;
                if (!caculatePoints(f7, f8, f5, f3, f, pointFArr9[0], pointFArr9[1])) {
                    Log.i(PlayView_Five.TAG, String.format(PlayView_Five.TAG, "startPt, criterion < 0, (%.2f, %.2f)-->(%.2f, %.2f), kLine - %.2f, kVertLine - %.2f, b - %.2f", Float.valueOf(this.mStartX), Float.valueOf(this.mStartY), Float.valueOf(this.mEndX), Float.valueOf(this.mEndY), Float.valueOf(f6), Float.valueOf(f7), Float.valueOf(f8)));
                    return false;
                }
                float f9 = this.mEndY;
                float f10 = this.mEndX;
                float f11 = f9 - (f7 * f10);
                PointF[] pointFArr10 = this.mPoints;
                boolean caculatePoints = caculatePoints(f7, f11, f10, f9, f, pointFArr10[2], pointFArr10[3]);
                float f12 = this.mStartX;
                if (!caculatePoints) {
                    Log.i(PlayView_Five.TAG, String.format(PlayView_Five.TAG, "endPt, criterion < 0, (%.2f, %.2f)-->(%.2f, %.2f), kLine - %.2f, kVertLine - %.2f, b - %.2f", Float.valueOf(f12), Float.valueOf(this.mStartY), Float.valueOf(this.mEndX), Float.valueOf(this.mEndY), Float.valueOf(f6), Float.valueOf(f7), Float.valueOf(f11)));
                    return false;
                }
                float f13 = this.mEndX;
                float f14 = this.mStartY;
                if (f12 < f13) {
                    float f15 = this.mEndY;
                    PointF[] pointFArr11 = this.mPoints;
                    if (f14 < f15) {
                        if (pointFArr11[0].x < this.mPoints[1].x) {
                            PointF[] pointFArr12 = this.mPoints;
                            swapPoint(pointFArr12[0], pointFArr12[1]);
                        }
                        if (this.mPoints[2].x > this.mPoints[3].x) {
                            PointF[] pointFArr13 = this.mPoints;
                            swapPoint(pointFArr13[2], pointFArr13[3]);
                        }
                    } else {
                        if (pointFArr11[0].x > this.mPoints[1].x) {
                            PointF[] pointFArr14 = this.mPoints;
                            swapPoint(pointFArr14[0], pointFArr14[1]);
                        }
                        if (this.mPoints[2].x < this.mPoints[3].x) {
                            PointF[] pointFArr15 = this.mPoints;
                            swapPoint(pointFArr15[2], pointFArr15[3]);
                        }
                    }
                } else {
                    float f16 = this.mEndY;
                    PointF[] pointFArr16 = this.mPoints;
                    if (f14 < f16) {
                        if (pointFArr16[0].x < this.mPoints[1].x) {
                            PointF[] pointFArr17 = this.mPoints;
                            swapPoint(pointFArr17[0], pointFArr17[1]);
                        }
                        if (this.mPoints[2].x > this.mPoints[3].x) {
                            PointF[] pointFArr18 = this.mPoints;
                            swapPoint(pointFArr18[2], pointFArr18[3]);
                        }
                    } else {
                        if (pointFArr16[0].x > this.mPoints[1].x) {
                            PointF[] pointFArr19 = this.mPoints;
                            swapPoint(pointFArr19[0], pointFArr19[1]);
                        }
                        if (this.mPoints[2].x < this.mPoints[3].x) {
                            PointF[] pointFArr20 = this.mPoints;
                            swapPoint(pointFArr20[2], pointFArr20[3]);
                        }
                    }
                }
            }
            return true;
        }

        public void updatePathWithStartPoints(PointF pointF, PointF pointF2) {
            this.mPath.reset();
            this.mPath.moveTo(pointF.x, pointF.y);
            this.mPath.lineTo(pointF2.x, pointF2.y);
            this.mPath.lineTo(this.mPoints[2].x, this.mPoints[2].y);
            this.mPath.lineTo(this.mPoints[3].x, this.mPoints[3].y);
            this.mPath.close();
        }
    }

    public PlayView_Five(Context context) {
        super(context);
        this.mCurrentLine = null;
        this.mLines = null;
        this.mLaserX = 0.0f;
        this.mLaserY = 0.0f;
        this.mPaint = new Paint();
        this.mWidth = 0;
        this.mHeight = 0;
        this.mElapsed = 0L;
        this.mStrokeWidth = 20.0f;
        this.mCircleRadius = 10.0f;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.view.PlayView_Five.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                PlayView_Five.this.invalidate();
                return false;
            }
        });
        initialize(context);
    }

    public PlayView_Five(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mCurrentLine = null;
        this.mLines = null;
        this.mLaserX = 0.0f;
        this.mLaserY = 0.0f;
        this.mPaint = new Paint();
        this.mWidth = 0;
        this.mHeight = 0;
        this.mElapsed = 0L;
        this.mStrokeWidth = 20.0f;
        this.mCircleRadius = 10.0f;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.view.PlayView_Five.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                PlayView_Five.this.invalidate();
                return false;
            }
        });
        initialize(context);
    }

    private void addToPaths(LineElement lineElement) {
        if (this.mLines == null) {
            this.mLines = new ArrayList();
        }
        this.mLines.add(lineElement);
    }

    private void adjustLasterPosition() {
        float f = this.mLaserX;
        float f2 = this.mCircleRadius;
        if (f - f2 < 0.0f) {
            this.mLaserX = f2;
        } else {
            float f3 = f + f2;
            int i = this.mWidth;
            if (f3 > i) {
                this.mLaserX = i - f2;
            }
        }
        float f4 = this.mLaserY;
        if (f4 - f2 < 0.0f) {
            this.mLaserY = f2;
            return;
        }
        float f5 = f4 + f2;
        int i2 = this.mHeight;
        if (f5 > i2) {
            this.mLaserY = i2 - f2;
        }
    }

    public static float convertDipToPx(Context context, float f) {
        return TypedValue.applyDimension(1, f, context.getResources().getDisplayMetrics());
    }

    private void initialize(Context context) {
        this.mStrokeWidth = convertDipToPx(context, 22.0f);
        this.mCircleRadius = convertDipToPx(context, 10.0f);
        this.mPaint.setARGB(255, 255, 0, 0);
        this.mPaint.setAntiAlias(true);
        this.mPaint.setStrokeWidth(0.0f);
        this.mPaint.setStyle(Paint.Style.FILL);
    }

    private boolean isValidLine(float f, float f2, float f3, float f4) {
        return Math.abs(f - f3) > 1.0f || Math.abs(f2 - f4) > 1.0f;
    }

    private void updateLaserPosition(float f, float f2) {
        this.mLaserX = f;
        this.mLaserY = f2;
        adjustLasterPosition();
    }

    private void updatePaths() {
        int size = this.mLines.size();
        if (size == 0) {
            return;
        }
        int i = 0;
        LineElement lineElement = null;
        while (i < size) {
            lineElement = this.mLines.get(i);
            if (lineElement.updatePathPoints()) {
                break;
            } else {
                i++;
            }
        }
        if (i == size) {
            this.mLines.clear();
            return;
        }
        while (true) {
            i--;
            if (i < 0) {
                break;
            } else {
                this.mLines.remove(0);
            }
        }
        lineElement.updatePath();
        int size2 = this.mLines.size();
        LineElement lineElement2 = null;
        for (int i2 = 1; i2 < size2; i2++) {
            LineElement lineElement3 = this.mLines.get(i2);
            if (lineElement3.updatePathPoints()) {
                if (lineElement2 == null) {
                    lineElement2 = this.mLines.get(i2 - 1);
                }
                lineElement3.updatePathWithStartPoints(lineElement2.mPoints[3], lineElement2.mPoints[2]);
                lineElement2 = null;
            } else {
                this.mLines.remove(i2);
                size2 = this.mLines.size();
            }
        }
    }

    public void compactPaths() {
        int size = this.mLines.size();
        int i = size - 1;
        if (size == 0) {
            return;
        }
        int i2 = 247;
        while (i >= 0) {
            LineElement lineElement = this.mLines.get(i);
            int alpha = lineElement.mPaint.getAlpha();
            if (alpha != 255) {
                int i3 = alpha - 8;
                if (i3 > 0 && lineElement.mPathWidth >= 1.0f) {
                    lineElement.setAlpha(i3);
                    i--;
                    i2 -= 8;
                }
                i++;
                break;
            }
            if (i2 > 0 && lineElement.mPathWidth >= 1.0f) {
                lineElement.setAlpha(i2);
                i--;
                i2 -= 8;
            }
            i++;
            break;
        }
        if (i >= size) {
            this.mLines = null;
        } else if (i >= 0) {
            this.mLines = this.mLines.subList(i, size);
        }
        long elapsedRealtime = (40 - SystemClock.elapsedRealtime()) + this.mElapsed;
        if (elapsedRealtime < 0) {
            elapsedRealtime = 0;
        }
        Message message = new Message();
        message.what = 1;
        this.mHandler.sendMessageDelayed(message, elapsedRealtime);
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        this.mElapsed = SystemClock.elapsedRealtime();
        if (this.mLines != null) {
            updatePaths();
            for (LineElement lineElement : this.mLines) {
                if (lineElement.mStartX >= 0.0f && lineElement.mEndY >= 0.0f && !lineElement.mPath.isEmpty()) {
                    canvas.drawPath(lineElement.mPath, lineElement.mPaint);
                }
            }
            compactPaths();
        }
        canvas.drawCircle(this.mLaserX, this.mLaserY, this.mCircleRadius, this.mPaint);
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        this.mWidth = i;
        this.mHeight = i2;
        adjustLasterPosition();
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        float x = motionEvent.getX();
        float y = motionEvent.getY();
        int action = motionEvent.getAction();
        if (action == 1) {
            if (isValidLine(this.mCurrentLine.mStartX, this.mCurrentLine.mStartY, x, y)) {
                this.mCurrentLine.mEndX = x;
                this.mCurrentLine.mEndY = y;
                addToPaths(this.mCurrentLine);
            }
            this.mCurrentLine = null;
            updateLaserPosition(x, y);
            invalidate();
            return true;
        }
        if (action == 0) {
            this.mLines = null;
            LineElement lineElement = new LineElement(this.mStrokeWidth);
            this.mCurrentLine = lineElement;
            lineElement.mStartX = x;
            this.mCurrentLine.mStartY = y;
            updateLaserPosition(x, y);
            return true;
        }
        if (action == 2 && isValidLine(this.mCurrentLine.mStartX, this.mCurrentLine.mStartY, x, y)) {
            this.mCurrentLine.mEndX = x;
            this.mCurrentLine.mEndY = y;
            addToPaths(this.mCurrentLine);
            LineElement lineElement2 = new LineElement(this.mStrokeWidth);
            this.mCurrentLine = lineElement2;
            lineElement2.mStartX = x;
            this.mCurrentLine.mStartY = y;
            updateLaserPosition(x, y);
        }
        if (this.mHandler.hasMessages(1)) {
            this.mHandler.removeMessages(1);
        }
        Message message = new Message();
        message.what = 1;
        this.mHandler.sendMessageDelayed(message, 0L);
        return true;
    }
}
