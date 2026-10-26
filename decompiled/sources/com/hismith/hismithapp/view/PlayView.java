package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import kotlinx.coroutines.scheduling.WorkQueueKt;

/* loaded from: classes3.dex */
public class PlayView extends View {
    private static final String TAG = "PlayView";
    private Bitmap bitmap;
    private Canvas canvas;
    private float[] mEndXs;
    private float[] mEndYs;
    private PointList[] mPointArr;
    private float[] mStartXs;
    private float[] mStartYs;
    private Paint paint;
    private Paint[] paintArr;
    private int pointRadius;
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

        public void add(int i, Point point) {
            this.mArray.add(i, point);
        }

        public void add(Point point) {
            this.mArray.add(point);
        }

        public void addFirst(Point point) {
            ArrayList arrayList = new ArrayList();
            this.mArray = arrayList;
            arrayList.add(point);
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

    public PlayView(Context context) {
        this(context, null);
    }

    public PlayView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public PlayView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.paintArr = new Paint[20];
        this.mPointArr = new PointList[20];
        this.mStartXs = new float[20];
        this.mStartYs = new float[20];
        this.mEndXs = new float[20];
        this.mEndYs = new float[20];
        this.pointRadius = dp2px(20);
        this.scaleRadius = 0.06f;
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

    protected int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, getResources().getDisplayMetrics());
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
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
        motionEvent.getAction();
        int i = 0;
        if (action == 0) {
            this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
            while (i < motionEvent.getPointerCount()) {
                this.paintArr[i] = initPaint();
                this.mStartXs[i] = motionEvent.getX(i);
                this.mStartYs[i] = motionEvent.getY(i);
                this.mPointArr[i].addFirst(new Point(this.mStartXs[i], this.mStartYs[i]));
                this.canvas.drawPoint(this.mStartXs[i], this.mStartYs[i], this.paintArr[i]);
                i++;
            }
        } else if (action == 1) {
            Log.d(TAG, "MotionEvent.ACTION_UP");
            this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
        } else if (action == 2) {
            this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
            for (int i2 = 0; i2 < motionEvent.getPointerCount(); i2++) {
                int pointerId = motionEvent.getPointerId(i2);
                this.mPointArr[pointerId].add(0, new Point(motionEvent.getX(i2), motionEvent.getY(i2)));
                int i3 = 0;
                while (i3 < this.mPointArr[pointerId].size() - 1) {
                    float f = 1.0f - (this.scaleRadius * i3);
                    if (f < 0.0f) {
                        break;
                    }
                    this.paintArr[pointerId].setStrokeWidth(this.pointRadius * f);
                    Canvas canvas = this.canvas;
                    float f2 = this.mPointArr[pointerId].get(i3).x;
                    float f3 = this.mPointArr[pointerId].get(i3).y;
                    i3++;
                    canvas.drawLine(f2, f3, this.mPointArr[pointerId].get(i3).x, this.mPointArr[pointerId].get(i3).y, this.paintArr[pointerId]);
                }
            }
        } else if (action == 5) {
            this.canvas.drawColor(0, PorterDuff.Mode.CLEAR);
            while (i < motionEvent.getPointerCount()) {
                this.paintArr[i] = initPaint();
                this.mStartXs[i] = motionEvent.getX(i);
                this.mStartYs[i] = motionEvent.getY(i);
                this.mPointArr[i].addFirst(new Point(this.mStartXs[i], this.mStartYs[i]));
                this.canvas.drawPoint(this.mStartXs[i], this.mStartYs[i], this.paintArr[i]);
                i++;
            }
        }
        invalidate();
        return true;
    }
}
