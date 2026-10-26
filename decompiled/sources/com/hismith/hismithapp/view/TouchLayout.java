package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class TouchLayout extends ViewGroup implements View.OnTouchListener {
    private int downX;
    private int downY;
    private boolean isFloat;
    private boolean isTop;
    private int mChildBottom;
    private int mChildHeight;
    private int mChildLeft;
    private int mChildRight;
    private int mChildTop;
    private View mChildView;
    private int mChildWidth;
    private int mHeight;
    private Paint mPaint;
    private float mScaleHeight;
    private float mScaleItemHeight;
    private int mScaleLongLine;
    private int mScaleMax;
    private int mScaleNum;
    private int mScaleShortLine;
    private float mStrokeWidth;
    private int mWidth;
    private OnScaleChangeListener onScaleChangeListener;

    public interface OnScaleChangeListener {
        void onScaleChange(int i);
    }

    public TouchLayout(Context context) {
        this(context, null);
    }

    public TouchLayout(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public TouchLayout(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mStrokeWidth = MeasureUtil.dip2px(getContext(), 1.5f);
        this.mScaleLongLine = MeasureUtil.dip2px(getContext(), 10.0f);
        this.mScaleShortLine = MeasureUtil.dip2px(getContext(), 5.0f);
        this.mScaleNum = 10;
        this.mScaleMax = 100;
        this.isTop = true;
        init();
    }

    private void init() {
        Paint paint = new Paint();
        this.mPaint = paint;
        paint.setAntiAlias(true);
        this.mPaint.setStyle(Paint.Style.FILL);
        this.mPaint.setStrokeWidth(this.mStrokeWidth);
        this.mPaint.setColor(-7829368);
    }

    @Override // android.view.ViewGroup, android.view.View
    protected void dispatchDraw(Canvas canvas) {
        super.dispatchDraw(canvas);
    }

    @Override // android.view.ViewGroup, android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        LogUtil.e("onLayout ");
        if (getChildCount() > 0) {
            this.mChildView = getChildAt(0);
        }
        View view = this.mChildView;
        if (view != null) {
            this.mChildWidth = view.getMeasuredWidth();
            int measuredHeight = this.mChildView.getMeasuredHeight();
            this.mChildHeight = measuredHeight;
            if (this.mChildRight == 0) {
                int i5 = this.mWidth;
                int i6 = this.mChildWidth;
                int i7 = (i5 - i6) / 2;
                this.mChildLeft = i7;
                int i8 = this.mHeight - measuredHeight;
                this.mChildTop = i8;
                this.mChildRight = i7 + i6;
                this.mChildBottom = i8 + measuredHeight;
            }
            LogUtil.e("onLayout mChildLeft:" + this.mChildLeft + ", mChildRight:" + this.mChildRight);
            this.mChildView.layout(this.mChildLeft, this.mChildTop, this.mChildRight, this.mChildBottom);
            float f = (this.mHeight - this.mChildHeight) * 1.0f;
            this.mScaleHeight = f;
            this.mScaleItemHeight = f / this.mScaleNum;
            this.mChildView.setOnTouchListener(this);
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        measureChildren(i, i2);
        setMeasuredDimension(getDefaultSize(getSuggestedMinimumWidth(), i), getDefaultSize(getSuggestedMinimumHeight(), i2));
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        int i5;
        super.onSizeChanged(i, i2, i3, i4);
        LogUtil.e("onSizeChanged w:" + i + ", h:" + i2 + ", lodw:" + i3 + ", olh:" + i4);
        this.mWidth = i;
        this.mHeight = i2;
        if (i3 != 0) {
            int i6 = this.mChildWidth;
            if (i3 == i6) {
                i5 = (i - i6) / 2;
                this.mChildLeft = i5;
            } else {
                i5 = (int) ((i - i6) * ((this.mChildLeft * 1.0f) / (i3 - i6)));
                this.mChildLeft = i5;
            }
            this.mChildRight = i5 + i6;
            LogUtil.e("onSizeChanged mChildLeft:" + this.mChildLeft + ", mChildRight:" + this.mChildRight);
        }
    }

    @Override // android.view.View.OnTouchListener
    public boolean onTouch(View view, MotionEvent motionEvent) {
        int action = motionEvent.getAction();
        if (action == 0) {
            this.downX = (int) motionEvent.getRawX();
            this.downY = (int) motionEvent.getRawY();
        } else if (action == 2) {
            int rawX = ((int) motionEvent.getRawX()) - this.downX;
            int rawY = ((int) motionEvent.getRawY()) - this.downY;
            int left = view.getLeft() + rawX;
            int bottom = view.getBottom() + rawY;
            int right = view.getRight() + rawX;
            int top = view.getTop() + rawY;
            if (left < 0) {
                right = view.getWidth();
                left = 0;
            }
            if (top < 0) {
                bottom = view.getHeight();
                top = 0;
            }
            int i = this.mWidth;
            if (right > i) {
                left = i - view.getWidth();
                right = i;
            }
            int i2 = this.mHeight;
            if (bottom > i2) {
                top = i2 - view.getHeight();
                bottom = i2;
            }
            view.layout(left, top, right, bottom);
            this.mChildLeft = left;
            this.mChildTop = top;
            this.mChildRight = right;
            this.mChildBottom = bottom;
            int i3 = this.mScaleMax;
            int i4 = i3 - ((int) ((top / this.mScaleHeight) * i3));
            OnScaleChangeListener onScaleChangeListener = this.onScaleChangeListener;
            if (onScaleChangeListener != null) {
                onScaleChangeListener.onScaleChange(i4);
            }
            this.downX = (int) motionEvent.getRawX();
            this.downY = (int) motionEvent.getRawY();
            view.postInvalidate();
        }
        invalidate();
        return true;
    }

    public void setOnScaleChangeListener(OnScaleChangeListener onScaleChangeListener) {
        this.onScaleChangeListener = onScaleChangeListener;
    }
}
