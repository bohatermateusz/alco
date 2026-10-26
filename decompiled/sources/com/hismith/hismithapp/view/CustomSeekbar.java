package com.hismith.hismithapp.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.MotionEvent;
import androidx.appcompat.widget.AppCompatSeekBar;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class CustomSeekbar extends AppCompatSeekBar {
    private float height;
    private boolean isTouch;
    private Paint mPaint;
    private int progressWidht;
    private int textColor;
    private int textY;
    private final int trackTextOffset;
    private float width;

    public CustomSeekbar(Context context) {
        this(context, null);
    }

    public CustomSeekbar(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public CustomSeekbar(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isTouch = true;
        this.textColor = -1;
        this.trackTextOffset = MeasureUtil.dp2px(getContext(), 8.0f);
        initType(attributeSet);
        init();
    }

    private int getTextHeight(String str, Paint paint) {
        Rect rect = new Rect();
        paint.getTextBounds(str, 0, str.length(), rect);
        return rect.height();
    }

    private int getTextWidth(String str, Paint paint) {
        Rect rect = new Rect();
        paint.getTextBounds(str, 0, str.length(), rect);
        return rect.width();
    }

    private void init() {
        Paint paint = new Paint();
        this.mPaint = paint;
        paint.setAntiAlias(true);
        this.mPaint.setStrokeWidth(4.0f);
        this.mPaint.setColor(this.textColor);
        this.mPaint.setTextSize(MeasureUtil.sp2px(getContext(), 14.0f));
        int textHeight = getTextHeight("0", this.mPaint) + this.trackTextOffset;
        int textWidth = getTextWidth("00", this.mPaint) / 2;
        setPadding(getPaddingLeft() + textWidth, getPaddingTop(), getPaddingRight() + textWidth, getPaddingBottom() + textHeight);
    }

    private void initType(AttributeSet attributeSet) {
        TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, R.styleable.CustomSeekbar);
        this.textColor = obtainStyledAttributes.getColor(R.styleable.CustomSeekbar_textColor, -1);
        obtainStyledAttributes.recycle();
    }

    public boolean isTouch() {
        return this.isTouch;
    }

    @Override // androidx.appcompat.widget.AppCompatSeekBar, android.widget.AbsSeekBar, android.widget.ProgressBar, android.view.View
    protected synchronized void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float f = this.progressWidht / 10.0f;
        for (int i = 0; i <= 10; i++) {
            canvas.drawText(i + "", ((i * f) + getPaddingLeft()) - (getTextWidth(i + "", this.mPaint) / 2.0f), this.textY, this.mPaint);
        }
    }

    @Override // android.widget.AbsSeekBar, android.widget.ProgressBar, android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        int textHeight = getTextHeight("0", this.mPaint) + this.trackTextOffset;
        Rect bounds = getProgressDrawable().getBounds();
        this.progressWidht = bounds.right - bounds.left;
        this.textY = bounds.bottom + textHeight;
    }

    @Override // android.widget.AbsSeekBar, android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        if (this.isTouch) {
            return super.onTouchEvent(motionEvent);
        }
        return false;
    }

    public void setTouch(boolean z) {
        this.isTouch = z;
    }
}
