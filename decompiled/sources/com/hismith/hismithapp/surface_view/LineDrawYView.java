package com.hismith.hismithapp.surface_view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class LineDrawYView extends View {
    private String[] YData;
    private Paint mEffectPaint;
    private int mHeight;
    private float mLineTextIntersticeWidth;
    private int mScaleBottom;
    private int mScaleBottomY;
    private int mScaleHeight;
    private int mScaleLeft;
    private int mScaleLeftX;
    private int mScaleRight;
    private int mScaleRightX;
    private int mScaleTop;
    private int mScaleTopY;
    private int mScaleWidth;
    private float mStrokeWidth;
    private int mTextHeight;
    private Paint mTextPaint;
    private float mTextSize;
    private int mWidth;
    private int mXScaleLongLine;
    private float mYHeight;
    private float mYItemHeight;
    private int mYItemNum;

    public LineDrawYView(Context context) {
        this(context, null);
    }

    public LineDrawYView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public LineDrawYView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mScaleTop = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleBottom = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleLeft = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleRight = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mYItemNum = 10;
        this.YData = new String[]{"100", "80", "60", "40", "20", "0"};
        this.mStrokeWidth = MeasureUtil.dip2px(getContext(), 2.0f);
        this.mTextSize = MeasureUtil.sp2px(getContext(), 12.0f);
        this.mLineTextIntersticeWidth = MeasureUtil.dip2px(getContext(), 5.0f);
        this.mXScaleLongLine = MeasureUtil.dip2px(getContext(), 10.0f);
        init();
    }

    private void drawYScale(Canvas canvas) {
        Canvas canvas2;
        if (canvas != null) {
            int i = 0;
            int i2 = 0;
            while (i <= this.mYItemNum) {
                float f = (this.mYItemHeight * i) + ((this.mHeight - this.mYHeight) / 2.0f);
                if (i % 2 == 0) {
                    canvas2 = canvas;
                    canvas2.drawLine(this.mScaleLeftX, f, this.mScaleRightX, f, this.mEffectPaint);
                    canvas2.drawText(this.YData[i2], this.mLineTextIntersticeWidth, f + (this.mTextHeight / 2.0f), this.mTextPaint);
                    i2++;
                } else {
                    canvas2 = canvas;
                }
                i++;
                canvas = canvas2;
            }
        }
    }

    private void init() {
        Paint paint = new Paint();
        this.mTextPaint = paint;
        paint.setAntiAlias(true);
        this.mTextPaint.setStyle(Paint.Style.FILL);
        this.mTextPaint.setTextSize(this.mTextSize);
        this.mTextPaint.setColor(Color.parseColor("#7FFFFFFF"));
        Paint paint2 = new Paint();
        this.mEffectPaint = paint2;
        paint2.setAntiAlias(true);
        this.mEffectPaint.setStyle(Paint.Style.STROKE);
        this.mEffectPaint.setStrokeWidth(this.mStrokeWidth);
        this.mEffectPaint.setColor(Color.parseColor("#18FFFFFF"));
        this.mEffectPaint.setPathEffect(new DashPathEffect(new float[]{MeasureUtil.dip2px(getContext(), 6.0f), MeasureUtil.dip2px(getContext(), 4.0f)}, 0.0f));
        this.mTextHeight = MeasureUtil.calcTextHeight(this.mTextPaint, "0");
        this.mScaleLeft = (int) (this.mScaleLeft + MeasureUtil.calcTextWidth(this.mTextPaint, "100") + (this.mLineTextIntersticeWidth * 2.0f));
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        drawYScale(canvas);
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.mWidth = View.MeasureSpec.getSize(i);
        int size = View.MeasureSpec.getSize(i2);
        this.mHeight = size;
        int i3 = this.mWidth;
        int i4 = this.mScaleLeft;
        int i5 = this.mScaleRight;
        this.mScaleWidth = (i3 - i4) - i5;
        int i6 = this.mScaleTop;
        int i7 = this.mScaleBottom;
        int i8 = (size - i6) - i7;
        this.mScaleHeight = i8;
        this.mScaleTopY = i6;
        this.mScaleBottomY = size - i7;
        this.mScaleLeftX = i4;
        this.mScaleRightX = i3 - i5;
        float f = (i8 - (this.mXScaleLongLine * 2)) - this.mTextHeight;
        this.mYHeight = f;
        this.mYItemHeight = f / this.mYItemNum;
    }
}
