package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.os.Handler;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;
import androidx.core.view.MotionEventCompat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import kotlinx.coroutines.scheduling.WorkQueueKt;

/* loaded from: classes3.dex */
public class PlayView_Three extends View {
    private static final String TAG = "PlayView";
    private Bitmap bitmap;
    private Canvas canvas;
    private Handler handler;
    private boolean isDraw;
    private float[] mEndXs;
    private float[] mEndYs;
    private PointList[] mPointArr;
    private float[] mStartXs;
    private float[] mStartYs;
    private Runnable myRunnable;
    private Paint paint;
    private Paint[] paintArr;
    private int pointRadius;
    private int pointSize;
    private Random random;
    private float scaleRadius;

    class Point {
        public float x;
        public float y;

        public Point(float f, float f2) {
            this.x = f;
            this.y = f2;
        }
    }

    class PointList {
        List<Point> mArray = new ArrayList();

        PointList() {
        }

        public void add(Point point) {
            this.mArray.add(0, point);
            if (this.mArray.size() > PlayView_Three.this.pointSize) {
                this.mArray.remove(PlayView_Three.this.pointSize);
            }
        }

        public Point get(int i) {
            return this.mArray.get(i);
        }

        public void remove(int i) {
            this.mArray.remove(i);
        }

        public int size() {
            return this.mArray.size();
        }
    }

    public PlayView_Three(Context context) {
        this(context, null);
    }

    public PlayView_Three(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public PlayView_Three(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.paintArr = new Paint[20];
        this.mPointArr = new PointList[20];
        this.mStartXs = new float[20];
        this.mStartYs = new float[20];
        this.mEndXs = new float[20];
        this.mEndYs = new float[20];
        this.pointRadius = dp2px(20);
        this.scaleRadius = 0.06f;
        this.pointSize = (int) ((1.0d / 0.06f) + 1.0d);
        this.isDraw = false;
        this.handler = new Handler();
        this.myRunnable = new Runnable() { // from class: com.hismith.hismithapp.view.PlayView_Three.1
            @Override // java.lang.Runnable
            public void run() {
                PlayView_Three playView_Three;
                int i2 = 0;
                PlayView_Three.this.isDraw = false;
                PlayView_Three.this.postInvalidate();
                while (true) {
                    int length = PlayView_Three.this.mPointArr.length;
                    playView_Three = PlayView_Three.this;
                    if (i2 >= length) {
                        break;
                    }
                    if (playView_Three.mPointArr[i2].size() > 1) {
                        PlayView_Three.this.mPointArr[i2].remove(PlayView_Three.this.mPointArr[i2].size() - 1);
                        PlayView_Three.this.isDraw = true;
                    }
                    i2++;
                }
                boolean z = playView_Three.isDraw;
                PlayView_Three playView_Three2 = PlayView_Three.this;
                if (z) {
                    playView_Three2.handler.postDelayed(PlayView_Three.this.myRunnable, 500L);
                } else {
                    playView_Three2.handler.removeCallbacks(PlayView_Three.this.myRunnable);
                }
            }
        };
        init();
    }

    private int getRandomColor() {
        int nextInt = this.random.nextInt(256);
        int nextInt2 = this.random.nextInt(256);
        int i = nextInt + nextInt2;
        int i2 = 381 - i;
        if (i2 <= 0) {
            i2 = 0;
        }
        return Color.rgb(nextInt, nextInt2, i < 256 ? rand(WorkQueueKt.MASK, 255) : rand(i2, 255));
    }

    private void init() {
        int i = 0;
        while (true) {
            PointList[] pointListArr = this.mPointArr;
            if (i >= pointListArr.length) {
                this.random = new Random();
                Paint paint = new Paint(4);
                this.paint = paint;
                paint.setStyle(Paint.Style.STROKE);
                this.paint.setStrokeWidth(this.pointRadius);
                this.paint.setColor(getRandomColor());
                this.paint.setAntiAlias(true);
                this.paint.setDither(true);
                this.paint.setStrokeJoin(Paint.Join.ROUND);
                this.paint.setStrokeCap(Paint.Cap.ROUND);
                return;
            }
            pointListArr[i] = new PointList();
            i++;
        }
    }

    private Paint initPaint() {
        Paint paint = new Paint(4);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(this.pointRadius);
        paint.setColor(getRandomColor());
        paint.setAntiAlias(true);
        paint.setDither(true);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        return paint;
    }

    private int rand(int i, int i2) {
        return (this.random.nextInt(i2) % ((i2 - i) + 1)) + i;
    }

    private void startThread() {
        this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
        this.handler.removeCallbacks(this.myRunnable);
        this.handler.post(this.myRunnable);
    }

    private void stopThread() {
        this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
        this.handler.removeCallbacks(this.myRunnable);
    }

    protected int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, getResources().getDisplayMetrics());
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawColor(0, PorterDuff.Mode.CLEAR);
        for (int i = 0; i < this.mPointArr.length; i++) {
            for (int i2 = 1; i2 < this.mPointArr[i].size(); i2++) {
                float f = i2;
                this.paintArr[i].setStrokeWidth(this.pointRadius * (1.0f - (this.scaleRadius * f)));
                Log.e(TAG, "x=" + this.mPointArr[i].get(i2).x + ", y=" + this.mPointArr[i].get(i2).y + ", j=" + i2 + " StrokeWidth=" + this.paintArr[i].getStrokeWidth() + ", scale=" + (1.0f - (this.scaleRadius * f)));
                int i3 = i2 - 1;
                this.canvas.drawLine(this.mPointArr[i].get(i3).x, this.mPointArr[i].get(i3).y, this.mPointArr[i].get(i2).x, this.mPointArr[i].get(i2).y, this.paintArr[i]);
            }
            Log.e(TAG, "==================================");
        }
        canvas.drawBitmap(this.bitmap, 0.0f, 0.0f, this.paint);
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.bitmap = Bitmap.createBitmap(getMeasuredWidth(), getMeasuredHeight(), Bitmap.Config.RGB_565);
        this.canvas = new Canvas(this.bitmap);
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int action = motionEvent.getAction() & 255;
        int action2 = (motionEvent.getAction() & MotionEventCompat.ACTION_POINTER_INDEX_MASK) >> 8;
        int i = 0;
        if (action == 0) {
            Log.e(TAG, "down第1个手指按下");
            this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
            while (i < motionEvent.getPointerCount()) {
                this.paintArr[i] = initPaint();
                this.mStartXs[i] = motionEvent.getX(i);
                this.mStartYs[i] = motionEvent.getY(i);
                this.mPointArr[i].add(new Point(this.mStartXs[i], this.mStartYs[i]));
                this.canvas.drawPoint(this.mStartXs[i], this.mStartYs[i], this.paintArr[i]);
                i++;
            }
        } else if (action == 1) {
            Log.d(TAG, "MotionEvent.ACTION_UP");
            startThread();
        } else if (action == 2) {
            while (i < motionEvent.getPointerCount()) {
                int pointerId = motionEvent.getPointerId(i);
                this.mPointArr[pointerId].add(new Point(motionEvent.getX(i), motionEvent.getY(i)));
                Log.e(TAG, "==================" + pointerId + "===================" + this.mPointArr[pointerId].size() + "==========" + this.pointSize);
                i++;
            }
        } else if (action == 5) {
            Log.e(TAG, "第" + (action2 + 1) + "个手指按下");
            this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
            while (i < motionEvent.getPointerCount()) {
                int i2 = i + 1;
                Log.e(TAG, "POINTER第" + i2 + "个手指按下");
                this.paintArr[i] = initPaint();
                this.mStartXs[i] = motionEvent.getX(i);
                this.mStartYs[i] = motionEvent.getY(i);
                this.mPointArr[i].add(new Point(this.mStartXs[i], this.mStartYs[i]));
                this.canvas.drawPoint(this.mStartXs[i], this.mStartYs[i], this.paintArr[i]);
                i = i2;
            }
        }
        invalidate();
        return true;
    }
}
