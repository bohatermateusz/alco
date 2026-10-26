package com.hismith.hismithapp.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.widget.ProgressBar;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class HorizontalProgressBarWithNumber extends ProgressBar {
    private static final int DEFAULT_CIRCLE_COLOR = -12627531;
    private static final int DEFAULT_CIRCLE_RADIUS = 2;
    private static final int DEFAULT_COLOR_UNREACHED_COLOR = -2894118;
    private static final int DEFAULT_HEIGHT_REACHED_PROGRESS_BAR = 2;
    private static final int DEFAULT_HEIGHT_UNREACHED_PROGRESS_BAR = 2;
    private static final int DEFAULT_TEXT_COLOR = -261935;
    private static final int DEFAULT_TEXT_SIZE = 10;
    protected static final int VISIBLE = 0;
    protected int mCircleColor;
    protected int mCircleRadius;
    protected boolean mIfDrawText;
    protected Paint mPaint;
    protected int mReachedBarColor;
    protected int mReachedProgressBarHeight;
    protected int mRealWidth;
    private RectF mRectF;
    protected int mTextColor;
    protected int mTextSize;
    protected int mUnReachedBarColor;
    protected int mUnReachedProgressBarHeight;
    protected int mViewCenterX;
    protected int mViewCenterY;

    public HorizontalProgressBarWithNumber(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public HorizontalProgressBarWithNumber(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mPaint = new Paint();
        this.mTextColor = DEFAULT_TEXT_COLOR;
        this.mTextSize = sp2px(10);
        this.mReachedProgressBarHeight = dp2px(2);
        this.mReachedBarColor = DEFAULT_TEXT_COLOR;
        this.mUnReachedProgressBarHeight = dp2px(2);
        this.mUnReachedBarColor = DEFAULT_COLOR_UNREACHED_COLOR;
        this.mCircleColor = DEFAULT_CIRCLE_COLOR;
        this.mCircleRadius = dp2px(2);
        this.mIfDrawText = true;
        obtainStyledAttributes(attributeSet);
        this.mPaint.setTextSize(this.mTextSize);
        this.mPaint.setColor(this.mTextColor);
        this.mPaint.setAntiAlias(true);
    }

    private void obtainStyledAttributes(AttributeSet attributeSet) {
        TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, R.styleable.HorizontalProgressBarWithNumber);
        this.mTextColor = obtainStyledAttributes.getColor(R.styleable.HorizontalProgressBarWithNumber_progress_text_color, DEFAULT_TEXT_COLOR);
        this.mTextSize = (int) obtainStyledAttributes.getDimension(R.styleable.HorizontalProgressBarWithNumber_progress_text_size, this.mTextSize);
        this.mCircleColor = obtainStyledAttributes.getColor(R.styleable.HorizontalProgressBarWithNumber_progress_circle_color, DEFAULT_CIRCLE_COLOR);
        this.mCircleRadius = (int) obtainStyledAttributes.getDimension(R.styleable.HorizontalProgressBarWithNumber_progress_circle_radius, this.mCircleRadius);
        this.mReachedBarColor = obtainStyledAttributes.getColor(R.styleable.HorizontalProgressBarWithNumber_progress_reached_color, this.mTextColor);
        this.mUnReachedBarColor = obtainStyledAttributes.getColor(R.styleable.HorizontalProgressBarWithNumber_progress_unreached_color, DEFAULT_COLOR_UNREACHED_COLOR);
        this.mReachedProgressBarHeight = (int) obtainStyledAttributes.getDimension(R.styleable.HorizontalProgressBarWithNumber_progress_reached_bar_height, this.mReachedProgressBarHeight);
        this.mUnReachedProgressBarHeight = (int) obtainStyledAttributes.getDimension(R.styleable.HorizontalProgressBarWithNumber_progress_unreached_bar_height, this.mUnReachedProgressBarHeight);
        if (obtainStyledAttributes.getInt(R.styleable.HorizontalProgressBarWithNumber_progress_text_visibility, 0) != 0) {
            this.mIfDrawText = false;
        }
        obtainStyledAttributes.recycle();
    }

    protected int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, getResources().getDisplayMetrics());
    }

    @Override // android.widget.ProgressBar, android.view.View
    protected synchronized void onDraw(Canvas canvas) {
        int i = ((this.mRealWidth - this.mUnReachedProgressBarHeight) / 2) - this.mCircleRadius;
        Paint paint = new Paint();
        paint.setAntiAlias(true);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(this.mUnReachedProgressBarHeight);
        paint.setColor(this.mUnReachedBarColor);
        canvas.drawCircle(this.mViewCenterX, this.mViewCenterY, i, paint);
        paint.setStrokeWidth(this.mReachedProgressBarHeight);
        paint.setColor(this.mReachedBarColor);
        int i2 = this.mReachedProgressBarHeight;
        int i3 = this.mCircleRadius;
        int i4 = this.mRealWidth;
        canvas.drawArc(new RectF((i2 / 2) + i3, (i2 / 2) + i3, i4 - ((i2 / 2) + i3), i4 - ((i2 / 2) + i3)), -90.0f, (getProgress() * 360) / getMax(), false, paint);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(this.mCircleColor);
        double d = i;
        float sin = (float) ((this.mReachedProgressBarHeight / 2) + this.mCircleRadius + i + (Math.sin((((getProgress() * 360) / getMax()) * 3.141592653589793d) / 180.0d) * d));
        float cos = (float) ((i + ((this.mReachedProgressBarHeight / 2) + this.mCircleRadius)) - (d * Math.cos((((getProgress() * 360) / getMax()) * 3.141592653589793d) / 180.0d)));
        canvas.drawCircle(sin, cos, this.mCircleRadius, paint);
        String str = getProgress() + "''";
        float measureText = this.mPaint.measureText(str);
        float descent = (this.mPaint.descent() + this.mPaint.ascent()) / 2.0f;
        this.mPaint.setColor(this.mTextColor);
        canvas.drawText(str, sin - (measureText / 2.0f), cos - descent, this.mPaint);
    }

    @Override // android.widget.ProgressBar, android.view.View
    protected synchronized void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        int measuredWidth = getMeasuredWidth();
        int measuredHeight = getMeasuredHeight();
        this.mViewCenterX = measuredWidth / 2;
        this.mViewCenterY = measuredHeight / 2;
        this.mRealWidth = (measuredWidth - getPaddingRight()) - getPaddingLeft();
        this.mRectF = new RectF(getPaddingLeft(), getPaddingTop(), this.mRealWidth, this.mRealWidth);
    }

    protected int sp2px(int i) {
        return (int) TypedValue.applyDimension(2, i, getResources().getDisplayMetrics());
    }
}
