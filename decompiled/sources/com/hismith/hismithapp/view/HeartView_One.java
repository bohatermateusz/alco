package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class HeartView_One extends View {
    private float height;
    private boolean isRun;
    private boolean isScale;
    private float mBitHeight;
    private float mBitWidth;
    private Bitmap mBitmap;
    private Context mContext;
    private long mHeartRefreshTime;
    private Thread mHeartThread;
    private Paint mPaint;
    private float scaleWidth;
    private float vwidth;
    private float width;

    public HeartView_One(Context context) {
        this(context, null);
    }

    public HeartView_One(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public HeartView_One(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isRun = false;
        this.mHeartRefreshTime = 16L;
        this.isScale = true;
        this.mContext = context;
        init();
    }

    private void init() {
        Paint paint = new Paint();
        this.mPaint = paint;
        paint.setAntiAlias(true);
        this.mBitmap = BitmapFactory.decodeResource(getResources(), R.mipmap.ic_play_heart);
        this.mBitWidth = r0.getWidth();
        this.mBitHeight = this.mBitmap.getHeight();
        float f = this.mBitWidth;
        this.vwidth = f;
        this.scaleWidth = f * 0.6f;
        this.isScale = true;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshRipple() {
    }

    private void startSync() {
        stopSync();
        this.isRun = true;
        Thread thread = new Thread() { // from class: com.hismith.hismithapp.view.HeartView_One.1
            @Override // java.lang.Thread, java.lang.Runnable
            public void run() {
                while (HeartView_One.this.isRun) {
                    try {
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                        HeartView_One.this.postInvalidate();
                        Thread.sleep(HeartView_One.this.mHeartRefreshTime);
                        HeartView_One.this.refreshRipple();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                        return;
                    }
                }
            }
        };
        this.mHeartThread = thread;
        thread.start();
    }

    private void stopSync() {
        if (this.mHeartThread == null) {
            return;
        }
        this.isRun = false;
        this.mHeartThread = null;
    }

    protected int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, getResources().getDisplayMetrics());
    }

    @Override // android.view.View
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        startSync();
    }

    @Override // android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        stopSync();
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float f = this.vwidth;
        if (f < this.scaleWidth && this.isScale) {
            this.isScale = false;
        } else if (f > this.mBitWidth && !this.isScale) {
            this.isScale = true;
        }
        if (this.isScale) {
            this.vwidth = f - 10.0f;
        } else {
            this.vwidth = f + 5.0f;
        }
        Bitmap decodeResource = BitmapFactory.decodeResource(getResources(), R.mipmap.ic_play_heart);
        float min = this.vwidth / Math.min(this.width, this.height);
        Matrix matrix = new Matrix();
        matrix.postScale(min, min);
        canvas.drawBitmap(Bitmap.createBitmap(decodeResource, 0, 0, decodeResource.getWidth(), decodeResource.getHeight(), matrix, true), (this.width - r0.getWidth()) / 2.0f, (this.height - r0.getHeight()) / 2.0f, (Paint) null);
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.width = getMeasuredWidth();
        this.height = getMeasuredHeight();
    }
}
