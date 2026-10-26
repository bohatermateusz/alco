package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class DoubleSliderView extends View {
    private static final int MAX_THUMB = 2;
    private static final int MAX_VALUE = 100;
    private static final int MIN_DIFFERENCE = 10;
    private static final int MIN_THUMB = 1;
    private static final int MIN_VALUE = 0;
    private static final int NONE = 0;
    private static final float TOUCH_THRESHOLD = 50.0f;
    private int activeThumb;
    private int lineColor;
    private float lineEndX;
    private float lineHeight;
    private Paint linePaint;
    private float lineStartX;
    private OnRangeChangeListener listener;
    private float mTextHeight;
    private int maxProgress;
    private int maxThumbColor;
    private Paint maxThumbPaint;
    private float maxThumbX;
    private int minProgress;
    private int minThumbColor;
    private Paint minThumbPaint;
    private float minThumbX;
    private float padding;
    private int progressColor;
    private Paint progressPaint;
    private Paint textPaint;
    private float thumbRadius;

    public interface OnRangeChangeListener {
        void onMaxValueChanged(int i);

        void onMinValueChanged(int i);

        void onRangeChangeEnd();

        void onRangeChangeStart();
    }

    public DoubleSliderView(Context context) {
        super(context);
        this.lineHeight = 15.0f;
        this.thumbRadius = 24.0f;
        this.padding = 24.0f;
        this.minProgress = 20;
        this.maxProgress = 80;
        this.minThumbX = 0.0f;
        this.maxThumbX = 0.0f;
        this.lineStartX = 0.0f;
        this.lineEndX = 0.0f;
        this.activeThumb = 0;
        init();
    }

    public DoubleSliderView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.lineHeight = 15.0f;
        this.thumbRadius = 24.0f;
        this.padding = 24.0f;
        this.minProgress = 20;
        this.maxProgress = 80;
        this.minThumbX = 0.0f;
        this.maxThumbX = 0.0f;
        this.lineStartX = 0.0f;
        this.lineEndX = 0.0f;
        this.activeThumb = 0;
        init();
    }

    public DoubleSliderView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.lineHeight = 15.0f;
        this.thumbRadius = 24.0f;
        this.padding = 24.0f;
        this.minProgress = 20;
        this.maxProgress = 80;
        this.minThumbX = 0.0f;
        this.maxThumbX = 0.0f;
        this.lineStartX = 0.0f;
        this.lineEndX = 0.0f;
        this.activeThumb = 0;
        init();
    }

    private int getActiveThumb(float f) {
        float abs = Math.abs(f - this.minThumbX);
        float abs2 = Math.abs(f - this.maxThumbX);
        if (abs >= abs2 || abs > 50.0f) {
            return abs2 <= 50.0f ? 2 : 0;
        }
        return 1;
    }

    private void init() {
        this.lineColor = getContext().getColor(R.color.white10);
        Paint paint = new Paint(1);
        this.linePaint = paint;
        paint.setColor(this.lineColor);
        this.linePaint.setStyle(Paint.Style.FILL);
        this.progressColor = getContext().getColor(R.color.white50);
        Paint paint2 = new Paint(1);
        this.progressPaint = paint2;
        paint2.setColor(this.progressColor);
        this.progressPaint.setStyle(Paint.Style.FILL);
        this.minThumbColor = getContext().getColor(R.color.play_mode_star_position_line_circle_mark);
        this.maxThumbColor = getContext().getColor(R.color.play_mode_vibrate_line_circle_mark);
        Paint paint3 = new Paint(1);
        this.minThumbPaint = paint3;
        paint3.setColor(this.minThumbColor);
        this.minThumbPaint.setStyle(Paint.Style.FILL);
        this.minThumbPaint.setShadowLayer(4.0f, 2.0f, 2.0f, Color.parseColor("#55000000"));
        Paint paint4 = new Paint(1);
        this.maxThumbPaint = paint4;
        paint4.setColor(this.maxThumbColor);
        this.maxThumbPaint.setStyle(Paint.Style.FILL);
        this.maxThumbPaint.setShadowLayer(4.0f, 2.0f, 2.0f, Color.parseColor("#55000000"));
        Paint paint5 = new Paint(1);
        this.textPaint = paint5;
        paint5.setColor(-1);
        this.textPaint.setTextSize(28.0f);
        this.textPaint.setTextAlign(Paint.Align.CENTER);
        this.textPaint.setFakeBoldText(true);
        Paint.FontMetrics fontMetrics = this.textPaint.getFontMetrics();
        this.mTextHeight = (fontMetrics.ascent + fontMetrics.descent) / 2.0f;
        setLayerType(1, null);
    }

    private void moveThumb(float f) {
        float f2 = this.lineStartX;
        if (f < f2) {
            f = f2;
        }
        float f3 = this.lineEndX;
        if (f > f3) {
            f = f3;
        }
        float f4 = ((f - f2) / (f3 - f2)) * 100.0f;
        if (f4 < 0.0f) {
            f4 = 0.0f;
        }
        float f5 = f4 <= 100.0f ? f4 : 100.0f;
        int i = this.activeThumb;
        if (i == 1) {
            int i2 = this.maxProgress;
            if (f5 > i2 - 10) {
                f5 = i2 - 10;
            }
            this.minProgress = (int) f5;
        } else if (i == 2) {
            int i3 = this.minProgress;
            if (f5 < i3 + 10) {
                f5 = i3 + 10;
            }
            this.maxProgress = (int) f5;
        }
        if (this.minProgress < 0) {
            this.minProgress = 0;
        }
        if (this.maxProgress > 100) {
            this.maxProgress = 100;
        }
        updateThumbPositions();
    }

    private void updateThumbPositions() {
        float f = this.lineStartX;
        float f2 = this.lineEndX;
        this.minThumbX = ((this.minProgress / 100.0f) * (f2 - f)) + f;
        this.maxThumbX = f + ((this.maxProgress / 100.0f) * (f2 - f));
    }

    public float getMaxProgress() {
        return this.maxProgress;
    }

    public float getMinProgress() {
        return this.minProgress;
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float height = getHeight() / 2.0f;
        float f = this.lineStartX;
        float f2 = this.lineHeight;
        RectF rectF = new RectF(f, height - (f2 / 2.0f), this.lineEndX, (f2 / 2.0f) + height);
        float f3 = this.lineHeight;
        canvas.drawRoundRect(rectF, f3 / 2.0f, f3 / 2.0f, this.linePaint);
        float f4 = this.minThumbX;
        float f5 = this.lineHeight;
        RectF rectF2 = new RectF(f4, height - (f5 / 2.0f), this.maxThumbX, (f5 / 2.0f) + height);
        float f6 = this.lineHeight;
        canvas.drawRoundRect(rectF2, f6 / 2.0f, f6 / 2.0f, this.progressPaint);
        canvas.drawCircle(this.minThumbX, height, this.thumbRadius, this.minThumbPaint);
        canvas.drawCircle(this.maxThumbX, height, this.thumbRadius, this.maxThumbPaint);
        this.textPaint.setColor(ViewCompat.MEASURED_STATE_MASK);
        canvas.drawText(this.minProgress + "", this.minThumbX, height - this.mTextHeight, this.textPaint);
        this.textPaint.setColor(-1);
        canvas.drawText(this.maxProgress + "", this.maxThumbX, height - this.mTextHeight, this.textPaint);
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        float f = this.padding;
        this.lineStartX = f;
        this.lineEndX = i - f;
        updateThumbPositions();
    }

    /* JADX WARN: Code restructure failed: missing block: B:7:0x0011, code lost:
    
        if (r1 != 3) goto L26;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        OnRangeChangeListener onRangeChangeListener;
        float x = motionEvent.getX();
        int action = motionEvent.getAction();
        if (action == 0) {
            int activeThumb = getActiveThumb(x);
            this.activeThumb = activeThumb;
            if (activeThumb != 0 && (onRangeChangeListener = this.listener) != null) {
                onRangeChangeListener.onRangeChangeStart();
            }
            return true;
        }
        if (action != 1) {
            if (action == 2) {
                if (this.activeThumb != 0) {
                    moveThumb(x);
                    invalidate();
                    return true;
                }
            }
            return super.onTouchEvent(motionEvent);
        }
        if (this.activeThumb != 0) {
            moveThumb(x);
            invalidate();
            OnRangeChangeListener onRangeChangeListener2 = this.listener;
            if (onRangeChangeListener2 != null) {
                int i = this.activeThumb;
                if (i == 1) {
                    onRangeChangeListener2.onMinValueChanged(this.minProgress);
                } else if (i == 2) {
                    onRangeChangeListener2.onMaxValueChanged(this.maxProgress);
                }
                this.listener.onRangeChangeEnd();
            }
            this.activeThumb = 0;
            return true;
        }
        return super.onTouchEvent(motionEvent);
    }

    public void setMinDifference(int i) {
        int i2 = this.maxProgress;
        int i3 = this.minProgress;
        if (i2 - i3 < i) {
            if (i3 + i <= 100) {
                this.maxProgress = i3 + i;
            } else if (i2 - i >= 0) {
                this.minProgress = i2 - i;
            }
        }
        updateThumbPositions();
        invalidate();
    }

    public void setOnRangeChangeListener(OnRangeChangeListener onRangeChangeListener) {
        this.listener = onRangeChangeListener;
    }

    public void setProgress(int i, int i2) {
        if (i < 0) {
            i = 0;
        }
        if (i2 > 100) {
            i2 = 100;
        }
        if (i2 - i < 10) {
            int i3 = i + 10;
            if (i3 <= 100) {
                i2 = i3;
            } else {
                int i4 = i2 - 10;
                if (i4 >= 0) {
                    i = i4;
                }
            }
        }
        this.minProgress = i;
        this.maxProgress = i2;
        updateThumbPositions();
        invalidate();
    }

    public void setProgressColor(int i) {
        this.progressColor = i;
        this.progressPaint.setColor(i);
        invalidate();
    }

    public void setThumbRadius(float f) {
        this.thumbRadius = f;
        this.padding = f;
        invalidate();
    }
}
