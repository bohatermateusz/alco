package com.hismith.hismithapp.view;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.animation.LinearInterpolator;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class TabView extends View {
    private boolean isFull;
    private ValueAnimator mAnimation;
    private int mColorIndicator;
    private int mColorTextNormal;
    private int mColorTextSelected;
    private int mCount;
    private int mCurIndex;
    private int mDownIndex;
    private int mDuration;
    private float mFactor;
    private int mHeight;
    private boolean mIsDragging;
    private int mLastIndex;
    private OnTabSelectedListener mOnTabSelectedListener;
    private float mPadding;
    private Paint mPaintA;
    private Paint mPaintB;
    private Paint mPaintTitle;
    private Paint mPaintTitleCur;
    private Rect mRect;
    private RectF mRectF;
    private float mRectRadius;
    private float mReservedPadding;
    private int mTextSize;
    private String[] mTitles;
    private int mTouchSlop;
    private float mTouchX;
    private float mTouchY;
    private int mWidth;
    private List<Float[]> mWidthList;

    public interface OnTabSelectedListener {
        void onTabSelected(int i);
    }

    public TabView(Context context) {
        this(context, null);
    }

    public TabView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public TabView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mDownIndex = 0;
        this.isFull = false;
        initTypedArray(context, attributeSet);
        init(context);
    }

    private float getTextHeight(Paint paint) {
        Paint.FontMetrics fontMetrics = paint.getFontMetrics();
        return (float) ((Math.ceil(fontMetrics.descent - fontMetrics.top) + 2.0d) / 2.0d);
    }

    private void init(Context context) {
        this.mTouchSlop = ViewConfiguration.get(context).getScaledTouchSlop();
        String[] strArr = this.mTitles;
        this.mCount = strArr != null ? strArr.length : 0;
        this.mRect = new Rect();
        this.mRectF = new RectF();
        Paint paint = new Paint(1);
        this.mPaintA = paint;
        paint.setColor(this.mColorIndicator);
        Paint paint2 = new Paint(1);
        this.mPaintB = paint2;
        paint2.setColor(0);
        Paint paint3 = new Paint(1);
        this.mPaintTitle = paint3;
        paint3.setTextSize(this.mTextSize);
        this.mPaintTitle.setTextAlign(Paint.Align.CENTER);
        this.mPaintTitle.setColor(this.mColorTextNormal);
        Paint paint4 = new Paint(1);
        this.mPaintTitleCur = paint4;
        paint4.setTextSize(this.mTextSize);
        this.mPaintTitleCur.setTextAlign(Paint.Align.CENTER);
        this.mPaintTitleCur.setColor(this.mColorTextSelected);
        ValueAnimator ofFloat = ValueAnimator.ofFloat(0.0f, 1.0f);
        this.mAnimation = ofFloat;
        ofFloat.setDuration(this.mDuration);
        this.mAnimation.setInterpolator(new LinearInterpolator());
        this.mAnimation.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.TabView.1
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                TabView.this.mFactor = ((Float) valueAnimator.getAnimatedValue()).floatValue();
                TabView.this.invalidate();
            }
        });
        this.mAnimation.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.view.TabView.2
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                if (TabView.this.mFactor != 1.0f || TabView.this.mOnTabSelectedListener == null) {
                    return;
                }
                TabView.this.mOnTabSelectedListener.onTabSelected(TabView.this.mCurIndex);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
            }
        });
    }

    private void initTypedArray(Context context, AttributeSet attributeSet) {
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.TabView);
        this.mTextSize = (int) obtainStyledAttributes.getDimension(R.styleable.TabView_tabv_textSize, TypedValue.applyDimension(1, 14.0f, getResources().getDisplayMetrics()));
        this.mColorIndicator = obtainStyledAttributes.getColor(R.styleable.TabView_tabv_colorIndicator, Color.parseColor("#FF4081"));
        this.mColorTextNormal = obtainStyledAttributes.getColor(R.styleable.TabView_tabv_colorTextNormal, Color.parseColor("#FF4081"));
        this.mColorTextSelected = obtainStyledAttributes.getColor(R.styleable.TabView_tabv_colorTextSelected, Color.parseColor("#FF4081"));
        this.mPadding = (int) obtainStyledAttributes.getDimension(R.styleable.TabView_tabv_padding, 2.0f);
        this.mReservedPadding = (int) obtainStyledAttributes.getDimension(R.styleable.TabView_tabv_paddingSide, -1.0f);
        this.mDuration = obtainStyledAttributes.getInteger(R.styleable.TabView_tabv_duration, 250);
        this.mRectRadius = obtainStyledAttributes.getDimension(R.styleable.TabView_tabv_radius, MeasureUtil.dip2px(getContext(), 18.0f));
        this.isFull = obtainStyledAttributes.getBoolean(R.styleable.TabView_tabv_full, false);
        obtainStyledAttributes.recycle();
    }

    private void startAnim() {
        stopAnim();
        ValueAnimator valueAnimator = this.mAnimation;
        if (valueAnimator != null) {
            valueAnimator.start();
        }
    }

    private void stopAnim() {
        ValueAnimator valueAnimator = this.mAnimation;
        if (valueAnimator != null) {
            valueAnimator.cancel();
        }
    }

    public int getmCurIndex() {
        return this.mCurIndex;
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (this.mCount <= 0) {
            return;
        }
        this.mRect.set(0, 0, this.mWidth, this.mHeight);
        this.mRectF.set(this.mRect);
        Rect rect = this.mRect;
        float f = this.mPadding;
        rect.set((int) f, (int) f, (int) (this.mWidth - f), (int) (this.mHeight - f));
        this.mRectF.set(this.mRect);
        RectF rectF = this.mRectF;
        float f2 = this.mRectRadius;
        canvas.drawRoundRect(rectF, f2, f2, this.mPaintB);
        float floatValue = this.mWidthList.get(this.mLastIndex)[0].floatValue();
        float floatValue2 = floatValue + ((this.mWidthList.get(this.mCurIndex)[0].floatValue() - floatValue) * this.mFactor);
        this.mRect.set((int) floatValue2, 0, (int) (floatValue2 + this.mWidthList.get(this.mCurIndex)[1].floatValue()), this.mHeight);
        this.mRectF.set(this.mRect);
        RectF rectF2 = this.mRectF;
        float f3 = this.mRectRadius;
        canvas.drawRoundRect(rectF2, f3, f3, this.mPaintA);
        int textHeight = (this.mHeight + ((int) getTextHeight(this.mPaintTitle))) / 2;
        for (int i = 0; i < this.mCount; i++) {
            canvas.drawText(this.mTitles[i], this.mWidthList.get(i)[0].floatValue() + (this.mWidthList.get(i)[1].floatValue() / 2.0f), textHeight, this.mPaintTitle);
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        if (this.mWidth == 0) {
            this.mWidth = View.MeasureSpec.getSize(i);
        }
        this.mHeight = View.MeasureSpec.getSize(i2);
        if (this.isFull) {
            int i3 = this.mWidth / this.mCount;
            List<Float[]> list = this.mWidthList;
            if (list == null) {
                this.mWidthList = new ArrayList();
            } else {
                list.clear();
            }
            for (int i4 = 0; i4 < this.mTitles.length; i4++) {
                this.mWidthList.add(new Float[]{Float.valueOf(i3 * i4), Float.valueOf(i3)});
                LogUtil.e("item:" + (MeasureUtil.calcTextWidth(this.mPaintTitle, this.mTitles[i4]) + (this.mReservedPadding * 2.0f)));
            }
        }
        setMeasuredDimension(this.mWidth, this.mHeight);
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        if (this.mCount > 0) {
            float f = this.mFactor;
            if (f == 0.0f || f == 1.0f) {
                motionEvent.getActionMasked();
                motionEvent.getActionIndex();
                float x = motionEvent.getX();
                float y = motionEvent.getY();
                int action = motionEvent.getAction();
                if (action != 0) {
                    if (action != 1) {
                        if (action == 2) {
                            if (!this.mIsDragging && (Math.abs(x - this.mTouchX) > this.mTouchSlop || Math.abs(y - this.mTouchY) > this.mTouchSlop)) {
                                this.mIsDragging = true;
                            }
                            return !this.mIsDragging;
                        }
                        if (action != 3) {
                            return super.onTouchEvent(motionEvent);
                        }
                    }
                    if (!this.mIsDragging && x >= 0.0f && x <= this.mWidth && y >= 0.0f && y <= this.mHeight) {
                        int i = 0;
                        while (true) {
                            if (i >= this.mWidthList.size()) {
                                i = 0;
                                break;
                            }
                            if (x < this.mWidthList.get(i)[0].floatValue() + this.mWidthList.get(i)[1].floatValue()) {
                                break;
                            }
                            i++;
                        }
                        int min = Math.min(Math.max(i, 0), this.mCount - 1);
                        int i2 = this.mDownIndex;
                        if (min == i2) {
                            this.mLastIndex = this.mCurIndex;
                            this.mCurIndex = i2;
                            LogUtil.e("mLastIndex:" + this.mLastIndex + ", mCurIndex:" + this.mCurIndex);
                            startAnim();
                            return true;
                        }
                    }
                    return false;
                }
                this.mTouchX = x;
                this.mTouchY = y;
                this.mDownIndex = 0;
                int i3 = 0;
                while (true) {
                    if (i3 >= this.mWidthList.size()) {
                        break;
                    }
                    if (x < this.mWidthList.get(i3)[0].floatValue() + this.mWidthList.get(i3)[1].floatValue()) {
                        this.mDownIndex = i3;
                        break;
                    }
                    i3++;
                }
                int max = Math.max(this.mDownIndex, 0);
                this.mDownIndex = max;
                int min2 = Math.min(max, this.mCount - 1);
                this.mDownIndex = min2;
                this.mIsDragging = false;
                if (min2 != this.mCurIndex) {
                    return true;
                }
            }
        }
        return false;
    }

    public void select(int i, boolean z) {
        int i2 = this.mCurIndex;
        if (i == i2) {
            return;
        }
        this.mLastIndex = i2;
        this.mCurIndex = i;
        if (z) {
            startAnim();
        } else {
            this.mFactor = 1.0f;
            invalidate();
        }
    }

    public void setOnTabSelectedListener(OnTabSelectedListener onTabSelectedListener) {
        this.mOnTabSelectedListener = onTabSelectedListener;
    }

    public void setTitle(String[] strArr) {
        if (strArr == null || strArr.length <= 0) {
            return;
        }
        this.mTitles = strArr;
        this.mCount = strArr.length;
        if (!this.isFull) {
            this.mWidth = 0;
            this.mWidthList = new ArrayList();
            for (int i = 0; i < this.mTitles.length; i++) {
                this.mWidthList.add(new Float[]{Float.valueOf(this.mWidth * 1.0f), Float.valueOf(MeasureUtil.calcTextWidth(this.mPaintTitle, this.mTitles[i]) + (this.mReservedPadding * 2.0f))});
                LogUtil.e("item:" + (MeasureUtil.calcTextWidth(this.mPaintTitle, this.mTitles[i]) + (this.mReservedPadding * 2.0f)));
                this.mWidth = (int) (this.mWidth + MeasureUtil.calcTextWidth(this.mPaintTitle, this.mTitles[i]) + (this.mReservedPadding * 2.0f));
            }
        }
        postInvalidate();
    }
}
