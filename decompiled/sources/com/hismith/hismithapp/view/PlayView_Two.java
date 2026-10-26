package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Handler;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import kotlinx.coroutines.scheduling.WorkQueueKt;

/* loaded from: classes3.dex */
public class PlayView_Two extends View {
    private List<Float[]> circles;
    private float f;
    private Handler handler;
    private Runnable myRunnable;
    private List<Paint> paintList;
    private Random random;
    private float strokeWidth;

    public PlayView_Two(Context context) {
        super(context, null);
        this.paintList = new ArrayList();
        this.handler = new Handler();
        this.circles = new ArrayList();
        this.f = 0.0f;
        this.strokeWidth = 5.0f;
        this.myRunnable = new Runnable() { // from class: com.hismith.hismithapp.view.PlayView_Two.1
            @Override // java.lang.Runnable
            public void run() {
                int size = PlayView_Two.this.circles.size();
                PlayView_Two playView_Two = PlayView_Two.this;
                if (size > 0) {
                    playView_Two.postInvalidate();
                    PlayView_Two.this.handler.postDelayed(PlayView_Two.this.myRunnable, 33L);
                } else {
                    playView_Two.postInvalidate();
                    PlayView_Two.this.handler.removeCallbacks(PlayView_Two.this.myRunnable);
                }
            }
        };
    }

    public PlayView_Two(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public PlayView_Two(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.paintList = new ArrayList();
        this.handler = new Handler();
        this.circles = new ArrayList();
        this.f = 0.0f;
        this.strokeWidth = 5.0f;
        this.myRunnable = new Runnable() { // from class: com.hismith.hismithapp.view.PlayView_Two.1
            @Override // java.lang.Runnable
            public void run() {
                int size = PlayView_Two.this.circles.size();
                PlayView_Two playView_Two = PlayView_Two.this;
                if (size > 0) {
                    playView_Two.postInvalidate();
                    PlayView_Two.this.handler.postDelayed(PlayView_Two.this.myRunnable, 33L);
                } else {
                    playView_Two.postInvalidate();
                    PlayView_Two.this.handler.removeCallbacks(PlayView_Two.this.myRunnable);
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
        this.random = new Random();
    }

    private Paint initPaint() {
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(getRandomColor());
        paint.setStrokeWidth(this.strokeWidth);
        return paint;
    }

    private int rand(int i, int i2) {
        return (this.random.nextInt(i2) % ((i2 - i) + 1)) + i;
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (this.circles.size() != 0) {
            int i = 0;
            while (i < this.circles.size()) {
                Float[] fArr = this.circles.get(i);
                Paint paint = this.paintList.get(i);
                if (fArr[2].floatValue() == 500.0f) {
                    this.paintList.remove(i);
                    this.circles.remove(i);
                } else {
                    i++;
                }
                canvas.drawCircle(fArr[0].floatValue(), fArr[1].floatValue(), fArr[2].floatValue(), paint);
                fArr[2] = Float.valueOf(fArr[2].floatValue() + 10.0f);
            }
        }
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int action = motionEvent.getAction();
        int i = action >> 8;
        int i2 = action & 255;
        if (i2 == 0 || i2 == 5) {
            this.paintList.add(initPaint());
            this.circles.add(new Float[]{Float.valueOf(motionEvent.getX(i)), Float.valueOf(motionEvent.getY(i)), Float.valueOf(this.f)});
            this.handler.removeCallbacks(this.myRunnable);
            this.handler.post(this.myRunnable);
        }
        return true;
    }
}
