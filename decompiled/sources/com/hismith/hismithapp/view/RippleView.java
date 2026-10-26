package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;
import androidx.core.internal.view.SupportMenu;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class RippleView extends View {
    private float alpha;
    private AnimThread at;
    private int color;
    private float fAlpha;
    private float fCount;
    private float fMaxRate;
    private float fRate;
    private float fVarRate;
    private boolean isRun;
    private float maxX;
    private float minX;
    private Paint paint;
    private Path path;
    private float rate;

    private class AnimThread extends Thread {
        private AnimThread() {
        }

        @Override // java.lang.Thread, java.lang.Runnable
        public void run() {
            while (RippleView.this.isRun && !Thread.currentThread().isInterrupted()) {
                RippleView rippleView = RippleView.this;
                RippleView.access$216(rippleView, rippleView.fVarRate);
                RippleView rippleView2 = RippleView.this;
                RippleView.access$424(rippleView2, rippleView2.fAlpha);
                if (RippleView.this.alpha <= 0.0f) {
                    RippleView rippleView3 = RippleView.this;
                    rippleView3.rate = rippleView3.fRate;
                    RippleView.this.alpha = 255.0f;
                }
                try {
                    Thread.sleep(10L);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                RippleView.this.postInvalidate();
            }
        }
    }

    public RippleView(Context context) {
        this(context, null);
    }

    public RippleView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public RippleView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.fRate = 1.0f;
        this.fMaxRate = 20.0f;
        this.fVarRate = 0.2f;
        this.fAlpha = 1.0f;
        this.fCount = 200.0f;
        this.rate = 1.0f;
        this.color = SupportMenu.CATEGORY_MASK;
        this.alpha = 255.0f;
        this.isRun = false;
        this.minX = 1080.0f;
        this.maxX = 0.0f;
        init();
    }

    static /* synthetic */ float access$216(RippleView rippleView, float f) {
        float f2 = rippleView.rate + f;
        rippleView.rate = f2;
        return f2;
    }

    static /* synthetic */ float access$424(RippleView rippleView, float f) {
        float f2 = rippleView.alpha - f;
        rippleView.alpha = f2;
        return f2;
    }

    private void init() {
        Paint paint = new Paint();
        this.paint = paint;
        paint.setAntiAlias(true);
        this.paint.setColor(this.color);
        this.paint.setAlpha((int) this.alpha);
        this.paint.setStyle(Paint.Style.FILL);
        this.paint.setStrokeWidth(2.0f);
        this.path = new Path();
    }

    @Override // android.view.View
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        this.isRun = true;
        this.at = new AnimThread();
    }

    @Override // android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        this.isRun = false;
        if (this.at != null) {
            this.at = null;
        }
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        this.paint.setColor(this.color);
        this.paint.setAlpha((int) this.alpha);
        this.path.reset();
        float measuredWidth = getMeasuredWidth() / 2;
        float measuredHeight = getMeasuredHeight() / 2;
        this.path.moveTo(measuredWidth, measuredHeight - (this.fRate * this.rate));
        for (double d = 0.0d; d <= 6.283185307179586d; d += 0.001d) {
            float sin = (float) (Math.sin(d) * 16.0d * Math.sin(d) * Math.sin(d));
            float cos = (float) ((((Math.cos(d) * 13.0d) - (Math.cos(d * 2.0d) * 5.0d)) - (Math.cos(3.0d * d) * 2.0d)) - Math.cos(4.0d * d));
            float f = this.rate;
            float f2 = measuredWidth - (sin * f);
            float f3 = measuredHeight - (cos * f);
            float f4 = this.minX;
            if (f2 < f4) {
                this.minX = Math.min(f2, f4);
                LogUtil.e("minX:" + this.minX + ", 最小值角度：" + Math.toDegrees(d));
            }
            float f5 = this.maxX;
            if (f2 > f5) {
                this.maxX = Math.max(f2, f5);
                LogUtil.e("maxX:" + this.maxX + ", 最大值角度：" + Math.toDegrees(d));
            }
            this.path.lineTo(f2, f3);
        }
        canvas.drawPath(this.path, this.paint);
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.fAlpha = 255.0f / this.fCount;
        LogUtil.e("fVarRate:" + this.fVarRate + ", fAlpha:" + this.fAlpha);
    }
}
