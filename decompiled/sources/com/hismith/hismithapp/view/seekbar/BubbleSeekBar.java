package com.hismith.hismithapp.view.seekbar;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.Property;
import android.util.SparseArray;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.LinearInterpolator;
import androidx.core.content.ContextCompat;
import androidx.core.internal.view.SupportMenu;
import com.google.android.material.badge.BadgeDrawable;
import com.hismith.hismithapp.R;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.math.BigDecimal;

/* loaded from: classes3.dex */
public class BubbleSeekBar extends View {
    static final int NONE = -1;
    float dx;
    private boolean isAlwaysShowBubble;
    private boolean isAutoAdjustSectionMark;
    private boolean isFloatType;
    private boolean isHideBubble;
    private boolean isRtl;
    private boolean isSeekBySection;
    private boolean isSeekStepSection;
    private boolean isShowProgressInFloat;
    private boolean isShowSectionMark;
    private boolean isShowSectionText;
    private boolean isShowThumbText;
    private boolean isThumbBitmap;
    private boolean isThumbOnDragging;
    private boolean isTouchToSeek;
    private boolean isTouchToSeekAnimEnd;
    private long mAlwaysShowBubbleDelay;
    private long mAnimDuration;
    private float mBubbleCenterRawSolidX;
    private float mBubbleCenterRawSolidY;
    private float mBubbleCenterRawX;
    private int mBubbleColor;
    private int mBubbleRadius;
    private int mBubbleTextColor;
    private int mBubbleTextSize;
    private BubbleView mBubbleView;
    private BubbleConfigBuilder mConfigBuilder;
    private float mDelta;
    private WindowManager.LayoutParams mLayoutParams;
    private float mLeft;
    private float mMax;
    private float mMin;
    private Paint mPaint;
    private int[] mPoint;
    private float mPreSecValue;
    private float mPreThumbCenterX;
    private float mProgress;
    private OnProgressChangedListener mProgressListener;
    private Rect mRectText;
    private float mRight;
    private int mSecondTrackColor;
    private int mSecondTrackSize;
    private int mSectionCount;
    private float mSectionOffset;
    private SparseArray<String> mSectionTextArray;
    private int mSectionTextColor;
    private int mSectionTextInterval;
    private int mSectionTextPosition;
    private int mSectionTextSize;
    private float mSectionValue;
    private int mTextSpace;
    private float mThumbCenterX;
    private int mThumbColor;
    private int mThumbRadius;
    private int mThumbRadiusOnDragging;
    private int mThumbTextColor;
    private int mThumbTextSize;
    private int mTrackColor;
    private float mTrackLength;
    private int mTrackSize;
    private WindowManager mWindowManager;
    private Bitmap thumbBitmap;
    private boolean triggerBubbleShowing;
    private boolean triggerSeekBySection;

    private class BubbleView extends View {
        private Paint mBubblePaint;
        private Path mBubblePath;
        private RectF mBubbleRectF;
        private String mProgressText;
        private Rect mRect;

        BubbleView(BubbleSeekBar bubbleSeekBar, Context context) {
            this(bubbleSeekBar, context, null);
        }

        BubbleView(BubbleSeekBar bubbleSeekBar, Context context, AttributeSet attributeSet) {
            this(context, attributeSet, 0);
        }

        BubbleView(Context context, AttributeSet attributeSet, int i) {
            super(context, attributeSet, i);
            this.mProgressText = "";
            Paint paint = new Paint();
            this.mBubblePaint = paint;
            paint.setAntiAlias(true);
            this.mBubblePaint.setTextAlign(Paint.Align.CENTER);
            this.mBubblePath = new Path();
            this.mBubbleRectF = new RectF();
            this.mRect = new Rect();
        }

        @Override // android.view.View
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            this.mBubblePath.reset();
            float measuredWidth = getMeasuredWidth() / 2.0f;
            float measuredHeight = getMeasuredHeight() - (BubbleSeekBar.this.mBubbleRadius / 3.0f);
            this.mBubblePath.moveTo(measuredWidth, measuredHeight);
            this.mBubblePath.lineTo(measuredWidth - (this.mBubbleRectF.left / 2.0f), this.mBubbleRectF.bottom);
            this.mBubblePath.lineTo(measuredWidth + (this.mBubbleRectF.left / 2.0f), this.mBubbleRectF.bottom);
            this.mBubblePath.close();
            this.mBubblePaint.setColor(BubbleSeekBar.this.mBubbleColor);
            canvas.drawRoundRect(this.mBubbleRectF, 10.0f, 10.0f, this.mBubblePaint);
            canvas.drawPath(this.mBubblePath, this.mBubblePaint);
            this.mBubblePaint.setTextSize(BubbleSeekBar.this.mBubbleTextSize);
            this.mBubblePaint.setColor(BubbleSeekBar.this.mBubbleTextColor);
            Paint paint = this.mBubblePaint;
            String str = this.mProgressText;
            paint.getTextBounds(str, 0, str.length(), this.mRect);
            Paint.FontMetrics fontMetrics = this.mBubblePaint.getFontMetrics();
            canvas.drawText(this.mProgressText, getMeasuredWidth() / 2.0f, ((this.mBubbleRectF.bottom - this.mBubbleRectF.top) + ((fontMetrics.descent - fontMetrics.ascent) / 2.0f)) - ((this.mBubbleRectF.bottom - measuredHeight) / 2.0f), this.mBubblePaint);
        }

        @Override // android.view.View
        protected void onMeasure(int i, int i2) {
            super.onMeasure(i, i2);
            setMeasuredDimension(BubbleSeekBar.this.mBubbleRadius * 3, BubbleSeekBar.this.mBubbleRadius * 3);
            this.mBubbleRectF.set((getMeasuredWidth() / 2.0f) - BubbleSeekBar.this.mBubbleRadius, BubbleSeekBar.this.mBubbleRadius, (getMeasuredWidth() / 2.0f) + BubbleSeekBar.this.mBubbleRadius, (BubbleSeekBar.this.mBubbleRadius * 2) + (BubbleSeekBar.this.mBubbleRadius / 3.0f));
        }

        void setProgressText(String str) {
            if (str == null || this.mProgressText.equals(str)) {
                return;
            }
            this.mProgressText = str;
            invalidate();
        }
    }

    public interface CustomSectionTextArray {
        SparseArray<String> onCustomize(int i, SparseArray<String> sparseArray);
    }

    public interface OnProgressChangedListener {
        void getProgressOnActionUp(BubbleSeekBar bubbleSeekBar, int i, float f);

        void getProgressOnFinally(BubbleSeekBar bubbleSeekBar, int i, float f, boolean z);

        void onProgressChanged(BubbleSeekBar bubbleSeekBar, int i, float f, boolean z);
    }

    public static abstract class OnProgressChangedListenerAdapter implements OnProgressChangedListener {
        @Override // com.hismith.hismithapp.view.seekbar.BubbleSeekBar.OnProgressChangedListener
        public void getProgressOnActionUp(BubbleSeekBar bubbleSeekBar, int i, float f) {
        }

        @Override // com.hismith.hismithapp.view.seekbar.BubbleSeekBar.OnProgressChangedListener
        public void getProgressOnFinally(BubbleSeekBar bubbleSeekBar, int i, float f, boolean z) {
        }

        @Override // com.hismith.hismithapp.view.seekbar.BubbleSeekBar.OnProgressChangedListener
        public void onProgressChanged(BubbleSeekBar bubbleSeekBar, int i, float f, boolean z) {
        }
    }

    @Retention(RetentionPolicy.SOURCE)
    public @interface TextPosition {
        public static final int BELOW_SECTION_MARK = 2;
        public static final int BOTTOM_SIDES = 1;
        public static final int SIDES = 0;
    }

    public BubbleSeekBar(Context context) {
        this(context, null);
    }

    public BubbleSeekBar(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public BubbleSeekBar(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mSectionTextPosition = -1;
        this.mSectionTextArray = new SparseArray<>();
        this.mPoint = new int[2];
        this.isTouchToSeekAnimEnd = true;
        this.isThumbBitmap = true;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.BubbleSeekBar, i, 0);
        this.mMin = obtainStyledAttributes.getFloat(R.styleable.BubbleSeekBar_bsb_min, 0.0f);
        this.mMax = obtainStyledAttributes.getFloat(R.styleable.BubbleSeekBar_bsb_max, 100.0f);
        this.mProgress = obtainStyledAttributes.getFloat(R.styleable.BubbleSeekBar_bsb_progress, this.mMin);
        this.isFloatType = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_is_float_type, false);
        this.mTrackSize = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_track_size, BubbleUtils.dp2px(2));
        this.mSecondTrackSize = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_second_track_size, this.mTrackSize + BubbleUtils.dp2px(2));
        this.mThumbRadius = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_thumb_radius, this.mSecondTrackSize + BubbleUtils.dp2px(2));
        this.mThumbRadiusOnDragging = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_thumb_radius_on_dragging, this.mSecondTrackSize * 2);
        this.mSectionCount = obtainStyledAttributes.getInteger(R.styleable.BubbleSeekBar_bsb_section_count, 10);
        this.mTrackColor = obtainStyledAttributes.getColor(R.styleable.BubbleSeekBar_bsb_track_color, ContextCompat.getColor(context, R.color.colorPrimary));
        this.mSecondTrackColor = obtainStyledAttributes.getColor(R.styleable.BubbleSeekBar_bsb_second_track_color, ContextCompat.getColor(context, R.color.colorAccent));
        this.mThumbColor = obtainStyledAttributes.getColor(R.styleable.BubbleSeekBar_bsb_thumb_color, this.mSecondTrackColor);
        this.isShowSectionText = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_show_section_text, false);
        this.mSectionTextSize = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_section_text_size, BubbleUtils.sp2px(14));
        this.mSectionTextColor = obtainStyledAttributes.getColor(R.styleable.BubbleSeekBar_bsb_section_text_color, this.mTrackColor);
        this.isSeekStepSection = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_seek_step_section, false);
        this.isSeekBySection = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_seek_by_section, false);
        int integer = obtainStyledAttributes.getInteger(R.styleable.BubbleSeekBar_bsb_section_text_position, -1);
        if (integer == 0) {
            this.mSectionTextPosition = 0;
        } else if (integer == 1) {
            this.mSectionTextPosition = 1;
        } else if (integer == 2) {
            this.mSectionTextPosition = 2;
        } else {
            this.mSectionTextPosition = -1;
        }
        this.mSectionTextInterval = obtainStyledAttributes.getInteger(R.styleable.BubbleSeekBar_bsb_section_text_interval, 1);
        this.isShowThumbText = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_show_thumb_text, false);
        this.mThumbTextSize = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_thumb_text_size, BubbleUtils.sp2px(14));
        this.mThumbTextColor = obtainStyledAttributes.getColor(R.styleable.BubbleSeekBar_bsb_thumb_text_color, this.mSecondTrackColor);
        this.mBubbleColor = obtainStyledAttributes.getColor(R.styleable.BubbleSeekBar_bsb_bubble_color, this.mSecondTrackColor);
        this.mBubbleTextSize = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_bubble_text_size, BubbleUtils.sp2px(14));
        this.mBubbleTextColor = obtainStyledAttributes.getColor(R.styleable.BubbleSeekBar_bsb_bubble_text_color, -1);
        this.isShowSectionMark = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_show_section_mark, false);
        this.isAutoAdjustSectionMark = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_auto_adjust_section_mark, false);
        this.isShowProgressInFloat = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_show_progress_in_float, false);
        int integer2 = obtainStyledAttributes.getInteger(R.styleable.BubbleSeekBar_bsb_anim_duration, -1);
        this.mAnimDuration = integer2 < 0 ? 200L : integer2;
        this.isTouchToSeek = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_touch_to_seek, false);
        this.isAlwaysShowBubble = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_always_show_bubble, false);
        int integer3 = obtainStyledAttributes.getInteger(R.styleable.BubbleSeekBar_bsb_always_show_bubble_delay, 0);
        this.mAlwaysShowBubbleDelay = integer3 < 0 ? 0L : integer3;
        this.isHideBubble = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_hide_bubble, false);
        this.isRtl = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_rtl, false);
        setEnabled(obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_android_enabled, isEnabled()));
        int dimensionPixelSize = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_thumb_width, BubbleUtils.dp2px(19));
        int dimensionPixelSize2 = obtainStyledAttributes.getDimensionPixelSize(R.styleable.BubbleSeekBar_bsb_thumb_height, BubbleUtils.dp2px(25));
        boolean z = obtainStyledAttributes.getBoolean(R.styleable.BubbleSeekBar_bsb_show_thumb_bitmap, false);
        this.isThumbBitmap = z;
        if (z) {
            this.thumbBitmap = BubbleUtils.imageScale(BitmapFactory.decodeResource(getResources(), obtainStyledAttributes.getResourceId(R.styleable.BubbleSeekBar_bsb_thumb_image, R.mipmap.trade_node_tips)), dimensionPixelSize, dimensionPixelSize2);
        }
        obtainStyledAttributes.recycle();
        Paint paint = new Paint();
        this.mPaint = paint;
        paint.setAntiAlias(true);
        this.mPaint.setStrokeCap(Paint.Cap.ROUND);
        this.mPaint.setTextAlign(Paint.Align.CENTER);
        this.mRectText = new Rect();
        this.mTextSpace = BubbleUtils.dp2px(2);
        initConfigByPriority();
        if (this.isHideBubble) {
            return;
        }
        this.mWindowManager = (WindowManager) context.getSystemService("window");
        BubbleView bubbleView = new BubbleView(this, context);
        this.mBubbleView = bubbleView;
        bubbleView.setProgressText(this.isShowProgressInFloat ? String.valueOf(getProgressFloat()) : String.valueOf(getProgress()));
        WindowManager.LayoutParams layoutParams = new WindowManager.LayoutParams();
        this.mLayoutParams = layoutParams;
        layoutParams.gravity = BadgeDrawable.TOP_START;
        this.mLayoutParams.width = -2;
        this.mLayoutParams.height = -2;
        this.mLayoutParams.format = -3;
        this.mLayoutParams.flags = 524328;
        BubbleUtils.isMIUI();
        this.mLayoutParams.type = 2;
        calculateRadiusOfBubble();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void autoAdjustSection() {
        ValueAnimator valueAnimator;
        float f = 0.0f;
        int i = 0;
        while (i <= this.mSectionCount) {
            float f2 = this.mSectionOffset;
            f = (i * f2) + this.mLeft;
            float f3 = this.mThumbCenterX;
            if (f <= f3 && f3 - f <= f2) {
                break;
            } else {
                i++;
            }
        }
        boolean z = BigDecimal.valueOf((double) this.mThumbCenterX).setScale(1, 4).floatValue() == f;
        AnimatorSet animatorSet = new AnimatorSet();
        if (z) {
            valueAnimator = null;
        } else {
            float f4 = this.mThumbCenterX;
            float f5 = f4 - f;
            float f6 = this.mSectionOffset;
            valueAnimator = f5 <= f6 / 2.0f ? ValueAnimator.ofFloat(f4, f) : ValueAnimator.ofFloat(f4, ((i + 1) * f6) + this.mLeft);
            valueAnimator.setInterpolator(new LinearInterpolator());
            valueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.5
                @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                    BubbleSeekBar.this.mThumbCenterX = ((Float) valueAnimator2.getAnimatedValue()).floatValue();
                    BubbleSeekBar bubbleSeekBar = BubbleSeekBar.this;
                    bubbleSeekBar.mProgress = bubbleSeekBar.calculateProgress();
                    if (BubbleSeekBar.this.isHideBubble || BubbleSeekBar.this.mBubbleView.getParent() == null) {
                        BubbleSeekBar.this.processProgress();
                    } else {
                        BubbleSeekBar bubbleSeekBar2 = BubbleSeekBar.this;
                        bubbleSeekBar2.mBubbleCenterRawX = bubbleSeekBar2.calculateCenterRawXofBubbleView();
                        BubbleSeekBar.this.mLayoutParams.x = (int) (BubbleSeekBar.this.mBubbleCenterRawX + 0.5f);
                        BubbleSeekBar.this.mWindowManager.updateViewLayout(BubbleSeekBar.this.mBubbleView, BubbleSeekBar.this.mLayoutParams);
                        BubbleView bubbleView = BubbleSeekBar.this.mBubbleView;
                        boolean z2 = BubbleSeekBar.this.isShowProgressInFloat;
                        BubbleSeekBar bubbleSeekBar3 = BubbleSeekBar.this;
                        bubbleView.setProgressText(z2 ? String.valueOf(bubbleSeekBar3.getProgressFloat()) : String.valueOf(bubbleSeekBar3.getProgress()));
                    }
                    BubbleSeekBar.this.invalidate();
                    if (BubbleSeekBar.this.mProgressListener != null) {
                        OnProgressChangedListener onProgressChangedListener = BubbleSeekBar.this.mProgressListener;
                        BubbleSeekBar bubbleSeekBar4 = BubbleSeekBar.this;
                        onProgressChangedListener.onProgressChanged(bubbleSeekBar4, bubbleSeekBar4.getProgress(), BubbleSeekBar.this.getProgressFloat(), true);
                    }
                }
            });
        }
        if (!this.isHideBubble) {
            ObjectAnimator ofFloat = ObjectAnimator.ofFloat(this.mBubbleView, (Property<BubbleView, Float>) View.ALPHA, this.isAlwaysShowBubble ? 1.0f : 0.0f);
            long j = this.mAnimDuration;
            if (z) {
                animatorSet.setDuration(j).play(ofFloat);
            } else {
                animatorSet.setDuration(j).playTogether(valueAnimator, ofFloat);
            }
        } else if (!z) {
            animatorSet.setDuration(this.mAnimDuration).playTogether(valueAnimator);
        }
        animatorSet.addListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.6
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
                if (!BubbleSeekBar.this.isHideBubble && !BubbleSeekBar.this.isAlwaysShowBubble) {
                    BubbleSeekBar.this.hideBubble();
                }
                BubbleSeekBar bubbleSeekBar = BubbleSeekBar.this;
                bubbleSeekBar.mProgress = bubbleSeekBar.calculateProgress();
                BubbleSeekBar.this.isThumbOnDragging = false;
                BubbleSeekBar.this.isTouchToSeekAnimEnd = true;
                BubbleSeekBar.this.invalidate();
            }

            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                if (!BubbleSeekBar.this.isHideBubble && !BubbleSeekBar.this.isAlwaysShowBubble) {
                    BubbleSeekBar.this.hideBubble();
                }
                BubbleSeekBar bubbleSeekBar = BubbleSeekBar.this;
                bubbleSeekBar.mProgress = bubbleSeekBar.calculateProgress();
                BubbleSeekBar.this.isThumbOnDragging = false;
                BubbleSeekBar.this.isTouchToSeekAnimEnd = true;
                BubbleSeekBar.this.invalidate();
                if (BubbleSeekBar.this.mProgressListener != null) {
                    OnProgressChangedListener onProgressChangedListener = BubbleSeekBar.this.mProgressListener;
                    BubbleSeekBar bubbleSeekBar2 = BubbleSeekBar.this;
                    onProgressChangedListener.getProgressOnFinally(bubbleSeekBar2, bubbleSeekBar2.getProgress(), BubbleSeekBar.this.getProgressFloat(), true);
                }
            }
        });
        animatorSet.start();
    }

    private float calThumbCxWhenSeekStepSection(float f) {
        float f2 = this.mLeft;
        if (f <= f2) {
            return f2;
        }
        float f3 = this.mRight;
        if (f >= f3) {
            return f3;
        }
        float f4 = 0.0f;
        int i = 0;
        while (i <= this.mSectionCount) {
            float f5 = this.mSectionOffset;
            f4 = (i * f5) + this.mLeft;
            if (f4 <= f && f - f4 <= f5) {
                break;
            }
            i++;
        }
        float f6 = f - f4;
        float f7 = this.mSectionOffset;
        return f6 <= f7 / 2.0f ? f4 : ((i + 1) * f7) + this.mLeft;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public float calculateCenterRawXofBubbleView() {
        boolean z = this.isRtl;
        float f = this.mBubbleCenterRawSolidX;
        return z ? f - ((this.mTrackLength * (this.mProgress - this.mMin)) / this.mDelta) : f + ((this.mTrackLength * (this.mProgress - this.mMin)) / this.mDelta);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public float calculateProgress() {
        float f;
        float f2;
        if (this.isRtl) {
            f = ((this.mRight - this.mThumbCenterX) * this.mDelta) / this.mTrackLength;
            f2 = this.mMin;
        } else {
            f = ((this.mThumbCenterX - this.mLeft) * this.mDelta) / this.mTrackLength;
            f2 = this.mMin;
        }
        return f + f2;
    }

    private void calculateRadiusOfBubble() {
        String float2String;
        String float2String2;
        this.mPaint.setTextSize(this.mBubbleTextSize);
        boolean z = this.isShowProgressInFloat;
        boolean z2 = this.isRtl;
        if (z) {
            float2String = float2String(z2 ? this.mMax : this.mMin);
        } else {
            boolean z3 = this.isFloatType;
            if (z2) {
                float f = this.mMax;
                float2String = z3 ? float2String(f) : String.valueOf((int) f);
            } else {
                float f2 = this.mMin;
                float2String = z3 ? float2String(f2) : String.valueOf((int) f2);
            }
        }
        this.mPaint.getTextBounds(float2String, 0, float2String.length(), this.mRectText);
        int width = (this.mRectText.width() + (this.mTextSpace * 2)) >> 1;
        boolean z4 = this.isShowProgressInFloat;
        boolean z5 = this.isRtl;
        if (z4) {
            float2String2 = float2String(z5 ? this.mMin : this.mMax);
        } else {
            boolean z6 = this.isFloatType;
            if (z5) {
                float f3 = this.mMin;
                float2String2 = z6 ? float2String(f3) : String.valueOf((int) f3);
            } else {
                float f4 = this.mMax;
                float2String2 = z6 ? float2String(f4) : String.valueOf((int) f4);
            }
        }
        this.mPaint.getTextBounds(float2String2, 0, float2String2.length(), this.mRectText);
        int width2 = (this.mRectText.width() + (this.mTextSpace * 2)) >> 1;
        int dp2px = BubbleUtils.dp2px(14);
        this.mBubbleRadius = dp2px;
        this.mBubbleRadius = Math.max(dp2px, Math.max(width, width2)) + this.mTextSpace;
    }

    private String float2String(float f) {
        return String.valueOf(formatFloat(f));
    }

    private float formatFloat(float f) {
        return BigDecimal.valueOf(f).setScale(1, 4).floatValue();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideBubble() {
        BubbleView bubbleView = this.mBubbleView;
        if (bubbleView == null) {
            return;
        }
        bubbleView.setVisibility(8);
        if (this.mBubbleView.getParent() != null) {
            this.mWindowManager.removeViewImmediate(this.mBubbleView);
        }
    }

    private void initConfigByPriority() {
        float f = this.mMin;
        float f2 = this.mMax;
        if (f == f2) {
            f = 0.0f;
            this.mMin = 0.0f;
            f2 = 100.0f;
            this.mMax = 100.0f;
        }
        if (f > f2) {
            this.mMax = f;
            this.mMin = f2;
            float f3 = f2;
            f2 = f;
            f = f3;
        }
        float f4 = this.mProgress;
        if (f4 < f) {
            this.mProgress = f;
            f4 = f;
        }
        if (f4 > f2) {
            this.mProgress = f2;
        }
        int i = this.mSectionCount;
        if (i <= 0) {
            i = 10;
            this.mSectionCount = 10;
        }
        float f5 = f2 - f;
        this.mDelta = f5;
        float f6 = f5 / i;
        this.mSectionValue = f6;
        if (f6 < 1.0f) {
            this.isFloatType = true;
        }
        if (this.isFloatType) {
            this.isShowProgressInFloat = true;
        }
        int i2 = this.mSectionTextPosition;
        if (i2 != -1) {
            this.isShowSectionText = true;
        }
        if (this.isShowSectionText) {
            if (i2 == -1) {
                this.mSectionTextPosition = 0;
                i2 = 0;
            }
            if (i2 == 2) {
                this.isShowSectionMark = true;
            }
        }
        if (this.mSectionTextInterval < 1) {
            this.mSectionTextInterval = 1;
        }
        initSectionTextArray();
        if (this.isSeekStepSection) {
            this.isSeekBySection = false;
            this.isAutoAdjustSectionMark = false;
        }
        if (this.isAutoAdjustSectionMark && !this.isShowSectionMark) {
            this.isAutoAdjustSectionMark = false;
        }
        if (this.isSeekBySection) {
            float f7 = this.mMin;
            this.mPreSecValue = f7;
            if (this.mProgress != f7) {
                this.mPreSecValue = this.mSectionValue;
            }
            this.isShowSectionMark = true;
            this.isAutoAdjustSectionMark = true;
        }
        if (this.isHideBubble) {
            this.isAlwaysShowBubble = false;
        }
        if (this.isAlwaysShowBubble) {
            setProgress(this.mProgress);
        }
        this.mThumbTextSize = (this.isFloatType || this.isSeekBySection || (this.isShowSectionText && this.mSectionTextPosition == 2)) ? this.mSectionTextSize : this.mThumbTextSize;
    }

    /* JADX WARN: Removed duplicated region for block: B:26:0x0050  */
    /* JADX WARN: Removed duplicated region for block: B:29:0x0055  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void initSectionTextArray() {
        float f;
        float f2;
        int i = 0;
        boolean z = this.mSectionTextPosition == 2;
        boolean z2 = this.mSectionTextInterval > 1 && this.mSectionCount % 2 == 0;
        while (true) {
            int i2 = this.mSectionCount;
            if (i > i2) {
                return;
            }
            boolean z3 = this.isRtl;
            if (z3) {
                float f3 = this.mMax;
                f = this.mSectionValue;
                f2 = f3 - (i * f);
            } else {
                float f4 = this.mMin;
                f = this.mSectionValue;
                f2 = f4 + (i * f);
            }
            if (z) {
                if (z2) {
                    if (i % this.mSectionTextInterval == 0) {
                        f2 = z3 ? this.mMax - (f * i) : this.mMin + (f * i);
                    }
                }
                this.mSectionTextArray.put(i, !this.isFloatType ? float2String(f2) : ((int) f2) + "");
            } else {
                if (i != 0 && i != i2) {
                }
                this.mSectionTextArray.put(i, !this.isFloatType ? float2String(f2) : ((int) f2) + "");
            }
            i++;
        }
    }

    private boolean isThumbTouched(MotionEvent motionEvent) {
        if (!isEnabled()) {
            return false;
        }
        float f = (this.mTrackLength / this.mDelta) * (this.mProgress - this.mMin);
        float f2 = this.isRtl ? this.mRight - f : this.mLeft + f;
        float measuredHeight = getMeasuredHeight() / 2.0f;
        return ((motionEvent.getX() - f2) * (motionEvent.getX() - f2)) + ((motionEvent.getY() - measuredHeight) * (motionEvent.getY() - measuredHeight)) <= (this.mLeft + ((float) BubbleUtils.dp2px(8))) * (this.mLeft + ((float) BubbleUtils.dp2px(8)));
    }

    private boolean isTrackTouched(MotionEvent motionEvent) {
        return isEnabled() && motionEvent.getX() >= ((float) getPaddingLeft()) && motionEvent.getX() <= ((float) (getMeasuredWidth() - getPaddingRight())) && motionEvent.getY() >= ((float) getPaddingTop()) && motionEvent.getY() <= ((float) (getMeasuredHeight() - getPaddingBottom()));
    }

    private void locatePositionInWindow() {
        Window window;
        getLocationInWindow(this.mPoint);
        Object parent = getParent();
        if (parent instanceof View) {
            View view = (View) parent;
            if (view.getMeasuredWidth() > 0) {
                int[] iArr = this.mPoint;
                iArr[0] = iArr[0] % view.getMeasuredWidth();
            }
        }
        boolean z = this.isRtl;
        int[] iArr2 = this.mPoint;
        if (z) {
            this.mBubbleCenterRawSolidX = (iArr2[0] + this.mRight) - (this.mBubbleView.getMeasuredWidth() / 2.0f);
        } else {
            this.mBubbleCenterRawSolidX = (iArr2[0] + this.mLeft) - (this.mBubbleView.getMeasuredWidth() / 2.0f);
        }
        this.mBubbleCenterRawX = calculateCenterRawXofBubbleView();
        float measuredHeight = this.mPoint[1] - this.mBubbleView.getMeasuredHeight();
        this.mBubbleCenterRawSolidY = measuredHeight;
        this.mBubbleCenterRawSolidY = measuredHeight - BubbleUtils.dp2px(24);
        if (BubbleUtils.isMIUI()) {
            this.mBubbleCenterRawSolidY -= BubbleUtils.dp2px(4);
        }
        Context context = getContext();
        if (!(context instanceof Activity) || (window = ((Activity) context).getWindow()) == null || (window.getAttributes().flags & 1024) == 0) {
            return;
        }
        Resources system = Resources.getSystem();
        this.mBubbleCenterRawSolidY += system.getDimensionPixelSize(system.getIdentifier("status_bar_height", "dimen", "android"));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public float processProgress() {
        float f = this.mProgress;
        if (this.isSeekBySection && this.triggerSeekBySection) {
            float f2 = this.mSectionValue / 2.0f;
            if (this.isTouchToSeek) {
                if (f != this.mMin && f != this.mMax) {
                    for (int i = 0; i <= this.mSectionCount; i++) {
                        float f3 = this.mSectionValue;
                        float f4 = i * f3;
                        if (f4 < f && f4 + f3 >= f) {
                            return f2 + f4 > f ? f4 : f4 + f3;
                        }
                    }
                }
            }
            float f5 = this.mPreSecValue;
            if (f >= f5) {
                if (f >= f2 + f5) {
                    float f6 = f5 + this.mSectionValue;
                    this.mPreSecValue = f6;
                    return f6;
                }
            } else if (f < f5 - f2) {
                float f7 = f5 - this.mSectionValue;
                this.mPreSecValue = f7;
                return f7;
            }
            return f5;
        }
        return f;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void showBubble() {
        BubbleView bubbleView = this.mBubbleView;
        if (bubbleView == null || bubbleView.getParent() != null) {
            return;
        }
        this.mLayoutParams.x = (int) (this.mBubbleCenterRawX + 0.5f);
        this.mLayoutParams.y = (int) (this.mBubbleCenterRawSolidY + 0.5f);
        this.mBubbleView.setAlpha(0.0f);
        this.mBubbleView.setVisibility(0);
        this.mBubbleView.animate().alpha(1.0f).setDuration(this.isTouchToSeek ? 0L : this.mAnimDuration).setListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.7
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                BubbleSeekBar.this.mWindowManager.addView(BubbleSeekBar.this.mBubbleView, BubbleSeekBar.this.mLayoutParams);
            }
        }).start();
        this.mBubbleView.setProgressText(this.isShowProgressInFloat ? String.valueOf(getProgressFloat()) : String.valueOf(getProgress()));
    }

    void config(BubbleConfigBuilder bubbleConfigBuilder) {
        this.mMin = bubbleConfigBuilder.min;
        this.mMax = bubbleConfigBuilder.max;
        this.mProgress = bubbleConfigBuilder.progress;
        this.isFloatType = bubbleConfigBuilder.floatType;
        this.mTrackSize = bubbleConfigBuilder.trackSize;
        this.mSecondTrackSize = bubbleConfigBuilder.secondTrackSize;
        this.mThumbRadius = bubbleConfigBuilder.thumbRadius;
        this.mThumbRadiusOnDragging = bubbleConfigBuilder.thumbRadiusOnDragging;
        this.mTrackColor = bubbleConfigBuilder.trackColor;
        this.mSecondTrackColor = bubbleConfigBuilder.secondTrackColor;
        this.mThumbColor = bubbleConfigBuilder.thumbColor;
        this.mSectionCount = bubbleConfigBuilder.sectionCount;
        this.isShowSectionMark = bubbleConfigBuilder.showSectionMark;
        this.isAutoAdjustSectionMark = bubbleConfigBuilder.autoAdjustSectionMark;
        this.isShowSectionText = bubbleConfigBuilder.showSectionText;
        this.mSectionTextSize = bubbleConfigBuilder.sectionTextSize;
        this.mSectionTextColor = bubbleConfigBuilder.sectionTextColor;
        this.mSectionTextPosition = bubbleConfigBuilder.sectionTextPosition;
        this.mSectionTextInterval = bubbleConfigBuilder.sectionTextInterval;
        this.isShowThumbText = bubbleConfigBuilder.showThumbText;
        this.mThumbTextSize = bubbleConfigBuilder.thumbTextSize;
        this.mThumbTextColor = bubbleConfigBuilder.thumbTextColor;
        this.isShowProgressInFloat = bubbleConfigBuilder.showProgressInFloat;
        this.mAnimDuration = bubbleConfigBuilder.animDuration;
        this.isTouchToSeek = bubbleConfigBuilder.touchToSeek;
        this.isSeekStepSection = bubbleConfigBuilder.seekStepSection;
        this.isSeekBySection = bubbleConfigBuilder.seekBySection;
        this.mBubbleColor = bubbleConfigBuilder.bubbleColor;
        this.mBubbleTextSize = bubbleConfigBuilder.bubbleTextSize;
        this.mBubbleTextColor = bubbleConfigBuilder.bubbleTextColor;
        this.isAlwaysShowBubble = bubbleConfigBuilder.alwaysShowBubble;
        this.mAlwaysShowBubbleDelay = bubbleConfigBuilder.alwaysShowBubbleDelay;
        this.isHideBubble = bubbleConfigBuilder.hideBubble;
        this.isRtl = bubbleConfigBuilder.rtl;
        initConfigByPriority();
        calculateRadiusOfBubble();
        OnProgressChangedListener onProgressChangedListener = this.mProgressListener;
        if (onProgressChangedListener != null) {
            onProgressChangedListener.onProgressChanged(this, getProgress(), getProgressFloat(), false);
            this.mProgressListener.getProgressOnFinally(this, getProgress(), getProgressFloat(), false);
        }
        this.mConfigBuilder = null;
        requestLayout();
    }

    public void correctOffsetWhenContainerOnScrolling() {
        if (this.isHideBubble) {
            return;
        }
        locatePositionInWindow();
        if (this.mBubbleView.getParent() != null) {
            if (!this.isAlwaysShowBubble) {
                postInvalidate();
                return;
            }
            this.mLayoutParams.y = (int) (this.mBubbleCenterRawSolidY + 0.5f);
            this.mWindowManager.updateViewLayout(this.mBubbleView, this.mLayoutParams);
        }
    }

    public BubbleConfigBuilder getConfigBuilder() {
        BubbleConfigBuilder bubbleConfigBuilder = this.mConfigBuilder;
        if (bubbleConfigBuilder == null) {
            bubbleConfigBuilder = new BubbleConfigBuilder(this);
            this.mConfigBuilder = bubbleConfigBuilder;
        }
        bubbleConfigBuilder.min = this.mMin;
        this.mConfigBuilder.max = this.mMax;
        this.mConfigBuilder.progress = this.mProgress;
        this.mConfigBuilder.floatType = this.isFloatType;
        this.mConfigBuilder.trackSize = this.mTrackSize;
        this.mConfigBuilder.secondTrackSize = this.mSecondTrackSize;
        this.mConfigBuilder.thumbRadius = this.mThumbRadius;
        this.mConfigBuilder.thumbRadiusOnDragging = this.mThumbRadiusOnDragging;
        this.mConfigBuilder.trackColor = this.mTrackColor;
        this.mConfigBuilder.secondTrackColor = this.mSecondTrackColor;
        this.mConfigBuilder.thumbColor = this.mThumbColor;
        this.mConfigBuilder.sectionCount = this.mSectionCount;
        this.mConfigBuilder.showSectionMark = this.isShowSectionMark;
        this.mConfigBuilder.autoAdjustSectionMark = this.isAutoAdjustSectionMark;
        this.mConfigBuilder.showSectionText = this.isShowSectionText;
        this.mConfigBuilder.sectionTextSize = this.mSectionTextSize;
        this.mConfigBuilder.sectionTextColor = this.mSectionTextColor;
        this.mConfigBuilder.sectionTextPosition = this.mSectionTextPosition;
        this.mConfigBuilder.sectionTextInterval = this.mSectionTextInterval;
        this.mConfigBuilder.showThumbText = this.isShowThumbText;
        this.mConfigBuilder.thumbTextSize = this.mThumbTextSize;
        this.mConfigBuilder.thumbTextColor = this.mThumbTextColor;
        this.mConfigBuilder.showProgressInFloat = this.isShowProgressInFloat;
        this.mConfigBuilder.animDuration = this.mAnimDuration;
        this.mConfigBuilder.touchToSeek = this.isTouchToSeek;
        this.mConfigBuilder.seekStepSection = this.isSeekStepSection;
        this.mConfigBuilder.seekBySection = this.isSeekBySection;
        this.mConfigBuilder.bubbleColor = this.mBubbleColor;
        this.mConfigBuilder.bubbleTextSize = this.mBubbleTextSize;
        this.mConfigBuilder.bubbleTextColor = this.mBubbleTextColor;
        this.mConfigBuilder.alwaysShowBubble = this.isAlwaysShowBubble;
        this.mConfigBuilder.alwaysShowBubbleDelay = this.mAlwaysShowBubbleDelay;
        this.mConfigBuilder.hideBubble = this.isHideBubble;
        this.mConfigBuilder.rtl = this.isRtl;
        return this.mConfigBuilder;
    }

    public float getMax() {
        return this.mMax;
    }

    public float getMin() {
        return this.mMin;
    }

    public OnProgressChangedListener getOnProgressChangedListener() {
        return this.mProgressListener;
    }

    public int getProgress() {
        return Math.round(processProgress());
    }

    public float getProgressFloat() {
        return formatFloat(processProgress());
    }

    @Override // android.view.View
    protected void onDetachedFromWindow() {
        hideBubble();
        super.onDetachedFromWindow();
    }

    /* JADX WARN: Code restructure failed: missing block: B:38:0x0249, code lost:
    
        if (r5 != r18.mMax) goto L87;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    protected void onDraw(Canvas canvas) {
        float f;
        float f2;
        Canvas canvas2;
        super.onDraw(canvas);
        float paddingLeft = getPaddingLeft();
        float measuredWidth = getMeasuredWidth() - getPaddingRight();
        float paddingTop = getPaddingTop() + this.mThumbRadiusOnDragging;
        if (this.isShowSectionText) {
            this.mPaint.setColor(this.mSectionTextColor);
            this.mPaint.setTextSize(this.mSectionTextSize);
            this.mPaint.getTextBounds("0123456789", 0, "0123456789".length(), this.mRectText);
            int i = this.mSectionTextPosition;
            if (i == 0) {
                float height = (this.mRectText.height() / 2.0f) + paddingTop;
                String str = this.mSectionTextArray.get(0);
                this.mPaint.getTextBounds(str, 0, str.length(), this.mRectText);
                canvas.drawText(str, (this.mRectText.width() / 2.0f) + paddingLeft, height, this.mPaint);
                paddingLeft += this.mRectText.width() + this.mTextSpace;
                String str2 = this.mSectionTextArray.get(this.mSectionCount);
                this.mPaint.getTextBounds(str2, 0, str2.length(), this.mRectText);
                canvas.drawText(str2, measuredWidth - ((this.mRectText.width() + 0.5f) / 2.0f), height, this.mPaint);
                measuredWidth -= this.mRectText.width() + this.mTextSpace;
            } else if (i >= 1) {
                String str3 = this.mSectionTextArray.get(0);
                this.mPaint.getTextBounds(str3, 0, str3.length(), this.mRectText);
                float height2 = this.mThumbRadiusOnDragging + paddingTop + this.mTextSpace + this.mRectText.height();
                float f3 = this.mLeft;
                if (this.mSectionTextPosition == 1) {
                    canvas.drawText(str3, f3, height2, this.mPaint);
                }
                String str4 = this.mSectionTextArray.get(this.mSectionCount);
                this.mPaint.getTextBounds(str4, 0, str4.length(), this.mRectText);
                float f4 = this.mRight;
                if (this.mSectionTextPosition == 1) {
                    canvas.drawText(str4, f4, height2, this.mPaint);
                }
                paddingLeft = f3;
                measuredWidth = f4;
            }
        } else if (this.isShowThumbText && this.mSectionTextPosition == -1) {
            paddingLeft = this.mLeft;
            measuredWidth = this.mRight;
        }
        boolean z = this.isShowSectionText;
        if ((!z && !this.isShowThumbText) || this.mSectionTextPosition == 0) {
            int i2 = this.mThumbRadiusOnDragging;
            paddingLeft += i2;
            measuredWidth -= i2;
        }
        float f5 = paddingLeft;
        float f6 = measuredWidth;
        char c = 2;
        boolean z2 = z && this.mSectionTextPosition == 2;
        if (z2 || this.isShowSectionMark) {
            this.mPaint.setTextSize(this.mSectionTextSize);
            this.mPaint.getTextBounds("0123456789", 0, "0123456789".length(), this.mRectText);
            float height3 = this.mRectText.height() + paddingTop + this.mThumbRadiusOnDragging + this.mTextSpace;
            BubbleUtils.dp2px(2);
            float abs = this.isRtl ? this.mRight - ((this.mTrackLength / this.mDelta) * Math.abs(this.mProgress - this.mMin)) : this.mLeft + ((this.mTrackLength / this.mDelta) * Math.abs(this.mProgress - this.mMin));
            int i3 = 0;
            while (i3 <= this.mSectionCount) {
                float f7 = (i3 * this.mSectionOffset) + f5;
                boolean z3 = this.isRtl;
                char c2 = c;
                Paint paint = this.mPaint;
                if (z3) {
                    paint.setColor(f7 <= abs ? this.mTrackColor : this.mSecondTrackColor);
                } else {
                    paint.setColor(f7 <= abs ? this.mSecondTrackColor : this.mTrackColor);
                }
                if (z2) {
                    this.mPaint.setColor(this.mSectionTextColor);
                    if (this.mSectionTextArray.get(i3, null) != null) {
                        canvas.drawText(this.mSectionTextArray.get(i3), f7, height3, this.mPaint);
                    }
                }
                i3++;
                c = c2;
            }
        }
        boolean z4 = this.isThumbOnDragging;
        if (!z4 || this.isAlwaysShowBubble) {
            boolean z5 = this.isRtl;
            float f8 = this.mTrackLength;
            if (z5) {
                this.mThumbCenterX = f6 - ((f8 / this.mDelta) * (this.mProgress - this.mMin));
            } else {
                this.mThumbCenterX = ((f8 / this.mDelta) * (this.mProgress - this.mMin)) + f5;
            }
        }
        if (this.isShowThumbText && !z4 && this.isTouchToSeekAnimEnd) {
            this.mPaint.setColor(this.mThumbTextColor);
            this.mPaint.setTextSize(this.mThumbTextSize);
            this.mPaint.getTextBounds("0123456789", 0, "0123456789".length(), this.mRectText);
            float height4 = this.mRectText.height() + paddingTop + this.mThumbRadiusOnDragging + this.mTextSpace;
            if (!this.isFloatType) {
                if (this.isShowProgressInFloat && this.mSectionTextPosition == 1) {
                    float f9 = this.mProgress;
                    if (f9 != this.mMin) {
                    }
                }
                canvas.drawText(String.valueOf(getProgress()), this.mThumbCenterX, height4, this.mPaint);
            }
            canvas.drawText(String.valueOf(getProgressFloat()), this.mThumbCenterX, height4, this.mPaint);
        }
        this.mPaint.setColor(this.mSecondTrackColor);
        this.mPaint.setStrokeWidth(this.mSecondTrackSize);
        boolean z6 = this.isRtl;
        float f10 = this.mThumbCenterX;
        if (z6) {
            canvas.drawLine(f6, paddingTop, f10, paddingTop, this.mPaint);
            f = f6;
            f2 = f5;
        } else {
            f = f6;
            f2 = f5;
            canvas.drawLine(f2, paddingTop, f10, paddingTop, this.mPaint);
        }
        this.mPaint.setColor(this.mTrackColor);
        this.mPaint.setStrokeWidth(this.mTrackSize);
        boolean z7 = this.isRtl;
        float f11 = f2;
        float f12 = this.mThumbCenterX;
        if (z7) {
            canvas2 = canvas;
            canvas2.drawLine(f12, paddingTop, f11, paddingTop, this.mPaint);
        } else {
            canvas2 = canvas;
            canvas2.drawLine(f12, paddingTop, f, paddingTop, this.mPaint);
        }
        this.mPaint.setColor(this.mThumbColor);
        if (this.isThumbBitmap) {
            canvas2.drawBitmap(this.thumbBitmap, this.mThumbCenterX - (this.thumbBitmap.getWidth() / 2), paddingTop - (this.thumbBitmap.getHeight() / 2), this.mPaint);
            return;
        }
        float f13 = this.mThumbCenterX;
        boolean z8 = this.isThumbOnDragging;
        canvas2.drawRect(f13 - (z8 ? this.mThumbRadiusOnDragging : this.mThumbRadius), paddingTop - ((z8 ? this.mThumbRadiusOnDragging : this.mThumbRadius) * 1.5f), f13 + (z8 ? this.mThumbRadiusOnDragging : this.mThumbRadius), ((z8 ? this.mThumbRadiusOnDragging : this.mThumbRadius) * 1.5f) + paddingTop, this.mPaint);
        this.mPaint.setColor(SupportMenu.CATEGORY_MASK);
        canvas2.drawCircle(this.mThumbCenterX, paddingTop, 2.0f, this.mPaint);
    }

    @Override // android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        if (this.isHideBubble) {
            return;
        }
        locatePositionInWindow();
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        int i3 = this.mThumbRadiusOnDragging * 2;
        if (this.isShowThumbText) {
            this.mPaint.setTextSize(this.mThumbTextSize);
            this.mPaint.getTextBounds("j", 0, 1, this.mRectText);
            i3 += this.mRectText.height();
        }
        if (this.isShowSectionText && this.mSectionTextPosition >= 1) {
            this.mPaint.setTextSize(this.mSectionTextSize);
            this.mPaint.getTextBounds("j", 0, 1, this.mRectText);
            i3 = Math.max(i3, (this.mThumbRadiusOnDragging * 2) + this.mRectText.height());
        }
        setMeasuredDimension(resolveSize(BubbleUtils.dp2px(180), i), i3 + (this.mTextSpace * 2));
        this.mLeft = getPaddingLeft() + this.mThumbRadiusOnDragging;
        this.mRight = (getMeasuredWidth() - getPaddingRight()) - this.mThumbRadiusOnDragging;
        if (this.isShowSectionText) {
            this.mPaint.setTextSize(this.mSectionTextSize);
            int i4 = this.mSectionTextPosition;
            if (i4 == 0) {
                String str = this.mSectionTextArray.get(0);
                this.mPaint.getTextBounds(str, 0, str.length(), this.mRectText);
                this.mLeft += this.mRectText.width() + this.mTextSpace;
                String str2 = this.mSectionTextArray.get(this.mSectionCount);
                this.mPaint.getTextBounds(str2, 0, str2.length(), this.mRectText);
                this.mRight -= this.mRectText.width() + this.mTextSpace;
            } else if (i4 >= 1) {
                String str3 = this.mSectionTextArray.get(0);
                this.mPaint.getTextBounds(str3, 0, str3.length(), this.mRectText);
                this.mLeft = getPaddingLeft() + Math.max(this.mThumbRadiusOnDragging, this.mRectText.width() / 2.0f) + this.mTextSpace;
                String str4 = this.mSectionTextArray.get(this.mSectionCount);
                this.mPaint.getTextBounds(str4, 0, str4.length(), this.mRectText);
                this.mRight = ((getMeasuredWidth() - getPaddingRight()) - Math.max(this.mThumbRadiusOnDragging, this.mRectText.width() / 2.0f)) - this.mTextSpace;
            }
        } else if (this.isShowThumbText && this.mSectionTextPosition == -1) {
            this.mPaint.setTextSize(this.mThumbTextSize);
            String str5 = this.mSectionTextArray.get(0);
            this.mPaint.getTextBounds(str5, 0, str5.length(), this.mRectText);
            this.mLeft = getPaddingLeft() + Math.max(this.mThumbRadiusOnDragging, this.mRectText.width() / 2.0f) + this.mTextSpace;
            String str6 = this.mSectionTextArray.get(this.mSectionCount);
            this.mPaint.getTextBounds(str6, 0, str6.length(), this.mRectText);
            this.mRight = ((getMeasuredWidth() - getPaddingRight()) - Math.max(this.mThumbRadiusOnDragging, this.mRectText.width() / 2.0f)) - this.mTextSpace;
        }
        float f = this.mRight - this.mLeft;
        this.mTrackLength = f;
        this.mSectionOffset = (f * 1.0f) / this.mSectionCount;
        if (this.isHideBubble) {
            return;
        }
        this.mBubbleView.measure(i, i2);
    }

    @Override // android.view.View
    protected void onRestoreInstanceState(Parcelable parcelable) {
        if (!(parcelable instanceof Bundle)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        Bundle bundle = (Bundle) parcelable;
        this.mProgress = bundle.getFloat("progress");
        super.onRestoreInstanceState(bundle.getParcelable("save_instance"));
        BubbleView bubbleView = this.mBubbleView;
        if (bubbleView != null) {
            bubbleView.setProgressText(this.isShowProgressInFloat ? String.valueOf(getProgressFloat()) : String.valueOf(getProgress()));
        }
        setProgress(this.mProgress);
    }

    @Override // android.view.View
    protected Parcelable onSaveInstanceState() {
        Bundle bundle = new Bundle();
        bundle.putParcelable("save_instance", super.onSaveInstanceState());
        bundle.putFloat("progress", this.mProgress);
        return bundle;
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        post(new Runnable() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.1
            @Override // java.lang.Runnable
            public void run() {
                BubbleSeekBar.this.requestLayout();
            }
        });
    }

    /* JADX WARN: Code restructure failed: missing block: B:7:0x000e, code lost:
    
        if (r0 != 3) goto L101;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked != 0) {
            if (actionMasked != 1) {
                if (actionMasked == 2) {
                    if (this.isThumbOnDragging) {
                        if (this.isSeekStepSection) {
                            float calThumbCxWhenSeekStepSection = calThumbCxWhenSeekStepSection(motionEvent.getX());
                            if (calThumbCxWhenSeekStepSection != this.mPreThumbCenterX) {
                                this.mPreThumbCenterX = calThumbCxWhenSeekStepSection;
                                this.mThumbCenterX = calThumbCxWhenSeekStepSection;
                            }
                        } else {
                            float x = motionEvent.getX() + this.dx;
                            this.mThumbCenterX = x;
                            float f = this.mLeft;
                            if (x < f) {
                                this.mThumbCenterX = f;
                                x = f;
                            }
                            float f2 = this.mRight;
                            if (x > f2) {
                                this.mThumbCenterX = f2;
                            }
                        }
                        this.mProgress = calculateProgress();
                        if (this.isHideBubble || this.mBubbleView.getParent() == null) {
                            processProgress();
                        } else {
                            float calculateCenterRawXofBubbleView = calculateCenterRawXofBubbleView();
                            this.mBubbleCenterRawX = calculateCenterRawXofBubbleView;
                            this.mLayoutParams.x = (int) (calculateCenterRawXofBubbleView + 0.5f);
                            this.mWindowManager.updateViewLayout(this.mBubbleView, this.mLayoutParams);
                            this.mBubbleView.setProgressText(this.isShowProgressInFloat ? String.valueOf(getProgressFloat()) : String.valueOf(getProgress()));
                        }
                        invalidate();
                        OnProgressChangedListener onProgressChangedListener = this.mProgressListener;
                        if (onProgressChangedListener != null) {
                            onProgressChangedListener.onProgressChanged(this, getProgress(), getProgressFloat(), true);
                        }
                    }
                }
            }
            getParent().requestDisallowInterceptTouchEvent(false);
            if (this.isAutoAdjustSectionMark) {
                if (this.isTouchToSeek) {
                    postDelayed(new Runnable() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.2
                        @Override // java.lang.Runnable
                        public void run() {
                            BubbleSeekBar.this.isTouchToSeekAnimEnd = false;
                            BubbleSeekBar.this.autoAdjustSection();
                        }
                    }, this.mAnimDuration);
                } else {
                    autoAdjustSection();
                }
            } else if (this.isThumbOnDragging || this.isTouchToSeek) {
                if (this.isHideBubble) {
                    animate().setDuration(this.mAnimDuration).setStartDelay((this.isThumbOnDragging || !this.isTouchToSeek) ? 0L : 300L).setListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.3
                        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
                        public void onAnimationCancel(Animator animator) {
                            BubbleSeekBar.this.isThumbOnDragging = false;
                            BubbleSeekBar.this.invalidate();
                        }

                        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
                        public void onAnimationEnd(Animator animator) {
                            BubbleSeekBar.this.isThumbOnDragging = false;
                            BubbleSeekBar.this.invalidate();
                        }
                    }).start();
                } else {
                    postDelayed(new Runnable() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.4
                        @Override // java.lang.Runnable
                        public void run() {
                            BubbleSeekBar.this.mBubbleView.animate().alpha(BubbleSeekBar.this.isAlwaysShowBubble ? 1.0f : 0.0f).setDuration(BubbleSeekBar.this.mAnimDuration).setListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.4.1
                                @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
                                public void onAnimationCancel(Animator animator) {
                                    if (!BubbleSeekBar.this.isAlwaysShowBubble) {
                                        BubbleSeekBar.this.hideBubble();
                                    }
                                    BubbleSeekBar.this.isThumbOnDragging = false;
                                    BubbleSeekBar.this.invalidate();
                                }

                                @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
                                public void onAnimationEnd(Animator animator) {
                                    if (!BubbleSeekBar.this.isAlwaysShowBubble) {
                                        BubbleSeekBar.this.hideBubble();
                                    }
                                    BubbleSeekBar.this.isThumbOnDragging = false;
                                    BubbleSeekBar.this.invalidate();
                                }
                            }).start();
                        }
                    }, this.mAnimDuration);
                }
            }
            OnProgressChangedListener onProgressChangedListener2 = this.mProgressListener;
            if (onProgressChangedListener2 != null) {
                onProgressChangedListener2.onProgressChanged(this, getProgress(), getProgressFloat(), true);
                this.mProgressListener.getProgressOnActionUp(this, getProgress(), getProgressFloat());
            }
        } else {
            performClick();
            getParent().requestDisallowInterceptTouchEvent(true);
            boolean isThumbTouched = isThumbTouched(motionEvent);
            this.isThumbOnDragging = isThumbTouched;
            if (isThumbTouched) {
                if (this.isSeekBySection && !this.triggerSeekBySection) {
                    this.triggerSeekBySection = true;
                }
                if (this.isAlwaysShowBubble && !this.triggerBubbleShowing) {
                    this.triggerBubbleShowing = true;
                }
                if (!this.isHideBubble) {
                    showBubble();
                }
                invalidate();
            } else if (this.isTouchToSeek && isTrackTouched(motionEvent)) {
                this.isThumbOnDragging = true;
                if (this.isSeekBySection && !this.triggerSeekBySection) {
                    this.triggerSeekBySection = true;
                }
                if (this.isAlwaysShowBubble) {
                    hideBubble();
                    this.triggerBubbleShowing = true;
                }
                if (this.isSeekStepSection) {
                    float calThumbCxWhenSeekStepSection2 = calThumbCxWhenSeekStepSection(motionEvent.getX());
                    this.mPreThumbCenterX = calThumbCxWhenSeekStepSection2;
                    this.mThumbCenterX = calThumbCxWhenSeekStepSection2;
                } else {
                    float x2 = motionEvent.getX();
                    this.mThumbCenterX = x2;
                    float f3 = this.mLeft;
                    if (x2 < f3) {
                        this.mThumbCenterX = f3;
                        x2 = f3;
                    }
                    float f4 = this.mRight;
                    if (x2 > f4) {
                        this.mThumbCenterX = f4;
                    }
                }
                this.mProgress = calculateProgress();
                if (!this.isHideBubble) {
                    this.mBubbleCenterRawX = calculateCenterRawXofBubbleView();
                    showBubble();
                }
                invalidate();
            }
            this.dx = this.mThumbCenterX - motionEvent.getX();
        }
        return this.isThumbOnDragging || this.isTouchToSeek || super.onTouchEvent(motionEvent);
    }

    @Override // android.view.View
    protected void onVisibilityChanged(View view, int i) {
        if (this.isHideBubble || !this.isAlwaysShowBubble) {
            return;
        }
        if (i != 0) {
            hideBubble();
        } else if (this.triggerBubbleShowing) {
            showBubble();
        }
        super.onVisibilityChanged(view, i);
    }

    @Override // android.view.View
    public boolean performClick() {
        return super.performClick();
    }

    public void setBubbleColor(int i) {
        if (this.mBubbleColor != i) {
            this.mBubbleColor = i;
            BubbleView bubbleView = this.mBubbleView;
            if (bubbleView != null) {
                bubbleView.invalidate();
            }
        }
    }

    public void setCustomSectionTextArray(CustomSectionTextArray customSectionTextArray) {
        this.mSectionTextArray = customSectionTextArray.onCustomize(this.mSectionCount, this.mSectionTextArray);
        for (int i = 0; i <= this.mSectionCount; i++) {
            if (this.mSectionTextArray.get(i) == null) {
                this.mSectionTextArray.put(i, "");
            }
        }
        this.isShowThumbText = false;
        requestLayout();
        invalidate();
    }

    public void setOnProgressChangedListener(OnProgressChangedListener onProgressChangedListener) {
        this.mProgressListener = onProgressChangedListener;
    }

    public void setProgress(float f) {
        this.mProgress = f;
        OnProgressChangedListener onProgressChangedListener = this.mProgressListener;
        if (onProgressChangedListener != null) {
            onProgressChangedListener.onProgressChanged(this, getProgress(), getProgressFloat(), false);
            this.mProgressListener.getProgressOnFinally(this, getProgress(), getProgressFloat(), false);
        }
        if (!this.isHideBubble) {
            this.mBubbleCenterRawX = calculateCenterRawXofBubbleView();
        }
        if (this.isAlwaysShowBubble) {
            hideBubble();
            postDelayed(new Runnable() { // from class: com.hismith.hismithapp.view.seekbar.BubbleSeekBar.8
                @Override // java.lang.Runnable
                public void run() {
                    BubbleSeekBar.this.showBubble();
                    BubbleSeekBar.this.triggerBubbleShowing = true;
                }
            }, this.mAlwaysShowBubbleDelay);
        }
        if (this.isSeekBySection) {
            this.triggerSeekBySection = false;
        }
        postInvalidate();
    }

    public void setSecondTrackColor(int i) {
        if (this.mSecondTrackColor != i) {
            this.mSecondTrackColor = i;
            invalidate();
        }
    }

    public void setThumbColor(int i) {
        if (this.mThumbColor != i) {
            this.mThumbColor = i;
            invalidate();
        }
    }

    public void setTrackColor(int i) {
        if (this.mTrackColor != i) {
            this.mTrackColor = i;
            invalidate();
        }
    }
}
