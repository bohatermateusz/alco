package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.widget.ProgressBar;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class TextProgressBar extends ProgressBar {
    private int mMaxTextSize;
    private int mMinTextSize;
    private String text;
    private Paint textPaint;

    public TextProgressBar(Context context) {
        this(context, null);
    }

    public TextProgressBar(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public TextProgressBar(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mMinTextSize = MeasureUtil.sp2px(getContext(), 8.0f);
        this.mMaxTextSize = MeasureUtil.sp2px(getContext(), 16.0f);
        initPaint();
    }

    private void initPaint() {
        Paint paint = new Paint();
        this.textPaint = paint;
        paint.setAntiAlias(true);
        this.textPaint.setColor(-1);
        this.textPaint.setTextSize(this.mMaxTextSize);
    }

    private void setText(int i) {
        this.text = ((int) (((i * 1.0f) / getMax()) * 100.0f)) + "%";
    }

    @Override // android.widget.ProgressBar, android.view.View
    protected synchronized void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Rect rect = new Rect();
        Paint paint = this.textPaint;
        String str = this.text;
        int i = 0;
        paint.getTextBounds(str, 0, str.length(), rect);
        int width = getWidth() / 2;
        rect.centerX();
        int height = (getHeight() / 2) - rect.centerY();
        int width2 = (((getWidth() * ((int) (((getProgress() * 1.0f) / getMax()) * 100.0f))) / 100) / 2) - rect.centerX();
        if (width2 >= 0) {
            i = width2;
        }
        canvas.drawText(this.text, i, height, this.textPaint);
    }

    @Override // android.widget.ProgressBar, android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        if (i2 == i4 || i2 <= 0) {
            return;
        }
        int paddingTop = (i2 - getPaddingTop()) - getPaddingBottom();
        float f = this.mMaxTextSize;
        while (this.textPaint.descent() - this.textPaint.ascent() > paddingTop) {
            f -= 1.0f;
            if (f < this.mMinTextSize) {
                return;
            } else {
                this.textPaint.setTextSize(f);
            }
        }
    }

    @Override // android.widget.ProgressBar
    public synchronized void setProgress(int i) {
        setText(i);
        super.setProgress(i);
    }
}
