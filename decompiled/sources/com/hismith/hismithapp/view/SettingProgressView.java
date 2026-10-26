package com.hismith.hismithapp.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class SettingProgressView extends View {
    private float mBigRadiusWidth;
    private boolean mCanDrag;
    private int mClickPosition;
    private int mColorSelected;
    private int mColorUnselected;
    private int mCurColor;
    private int mCurPosition;
    private float mDivX;
    private float mDivY;
    private float mFirstItemRange;
    private int mHeight;
    private boolean mIsClickValid;
    private int mItemCount;
    private float mItemWidth;
    private float mLastX;
    private float mLastY;
    private float mLineHeight;
    private OnProgressChangeListener mListener;
    private Paint mPaint;
    private Paint mPaintShader;
    private Path mPath;
    private float mRadiusBig;
    private float mRadiusSmall;
    private float mRadiusSpace;
    private float mShadeRadius;
    private Paint mTextPaint;
    private int mTouchSlop;
    private int mWidth;

    public interface OnProgressChangeListener {
        void onClick(int i);

        void onProgressChange(int i);
    }

    public SettingProgressView(Context context) {
        super(context, null);
    }

    public SettingProgressView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public SettingProgressView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.lib_ui_view_SettingProgressView);
        this.mItemCount = obtainStyledAttributes.getInteger(R.styleable.lib_ui_view_SettingProgressView_lib_ui_view_spv_count, 5);
        this.mColorSelected = obtainStyledAttributes.getColor(R.styleable.lib_ui_view_SettingProgressView_lib_ui_view_spv_colorSelect, Color.parseColor("#69B068"));
        this.mColorUnselected = obtainStyledAttributes.getColor(R.styleable.lib_ui_view_SettingProgressView_lib_ui_view_spv_colorUnselect, -7829368);
        this.mRadiusSmall = obtainStyledAttributes.getDimension(R.styleable.lib_ui_view_SettingProgressView_lib_ui_view_spv_radiusSmall, MeasureUtil.dp2px(context, 5.0f));
        this.mRadiusBig = obtainStyledAttributes.getDimension(R.styleable.lib_ui_view_SettingProgressView_lib_ui_view_spv_radiusBig, MeasureUtil.dp2px(context, 10.0f));
        this.mRadiusSpace = obtainStyledAttributes.getDimension(R.styleable.lib_ui_view_SettingProgressView_lib_ui_view_spv_radiusSpace, MeasureUtil.dp2px(context, 2.0f));
        obtainStyledAttributes.recycle();
        init(context);
    }

    private void drawCircle(Canvas canvas, int i, float f, float f2) {
        this.mPaint.setColor(i);
        canvas.drawCircle(f, f2, this.mRadiusSmall + this.mRadiusSpace, this.mPaint);
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

    private void init(Context context) {
        setLayerType(1, null);
        this.mTouchSlop = ViewConfiguration.get(getContext()).getScaledTouchSlop();
        this.mLineHeight = MeasureUtil.dp2px(context, 2.0f);
        this.mShadeRadius = MeasureUtil.dp2px(context, 5.0f);
        this.mPath = new Path();
        this.mPaint = new Paint(1);
        this.mPaintShader = new Paint(1);
        this.mTextPaint = new Paint(1);
        boolean equals = "eropair".equals("");
        Paint paint = this.mTextPaint;
        if (equals) {
            paint.setColor(ViewCompat.MEASURED_STATE_MASK);
        } else {
            paint.setColor(-1);
        }
        this.mTextPaint.setTextSize(MeasureUtil.sp2px(getContext(), 14.0f));
        this.mCurColor = -1;
        this.mPaintShader.setColor(-1);
        this.mPaintShader.setShadowLayer(this.mShadeRadius, 0.0f, 0.0f, this.mColorUnselected);
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        this.mPath.reset();
        this.mPaint.setColor(this.mColorUnselected);
        int textHeight = getTextHeight("0", this.mTextPaint);
        float f = this.mWidth;
        float f2 = this.mRadiusSpace;
        float f3 = this.mRadiusBig;
        float f4 = (f - ((f2 + f3) * 2.0f)) / (this.mItemCount - 1);
        this.mItemWidth = f4;
        float f5 = f2 + f3;
        float f6 = f2 + f3;
        float f7 = f6 + this.mLineHeight;
        float f8 = this.mCanDrag ? this.mDivX : (f4 * this.mCurPosition) + f2 + f3;
        canvas.drawRect(f5, f6, f8, f7, this.mPaint);
        this.mPaint.setColor(this.mColorSelected);
        this.mPath.addRect(f8, f6, (this.mWidth - this.mRadiusSpace) - this.mRadiusBig, f7, Path.Direction.CW);
        float f9 = f6 + (this.mLineHeight / 2.0f);
        for (int i = 0; i < this.mItemCount; i++) {
            float f10 = this.mRadiusBig;
            float f11 = this.mRadiusSpace;
            float f12 = f10 + f11 + (this.mItemWidth * i);
            int i2 = this.mCurPosition;
            if (i == i2) {
                if (this.mCanDrag) {
                    float f13 = this.mDivX;
                    if (f13 > f12) {
                        drawCircle(canvas, this.mColorUnselected, f12, f9);
                    } else if (f13 < f12) {
                        this.mPath.addCircle(f12, f9, this.mRadiusSmall + f11, Path.Direction.CW);
                    }
                    this.mPaint.setColor(this.mColorSelected);
                }
            } else if (i < i2) {
                drawCircle(canvas, this.mColorUnselected, f12, f9);
            } else {
                this.mPath.addCircle(f12, f9, this.mRadiusSmall + f11, Path.Direction.CW);
            }
            canvas.drawText(Constans.Slide_Speed_Arr[i] + "", f12 - (getTextWidth(Constans.Slide_Speed_Arr[i] + "", this.mTextPaint) / 2.0f), (f9 * 2.0f) + textHeight, this.mTextPaint);
        }
        this.mPaint.setColor(this.mColorSelected);
        canvas.drawPath(this.mPath, this.mPaint);
        for (int i3 = 0; i3 < this.mItemCount; i3++) {
            float f14 = this.mRadiusBig;
            float f15 = this.mRadiusSpace;
            float f16 = f14 + f15 + (this.mItemWidth * i3);
            if (i3 == this.mCurPosition) {
                if (this.mCanDrag) {
                    canvas.drawCircle(this.mDivX, f9, (f14 + f15) - this.mShadeRadius, this.mPaintShader);
                } else {
                    canvas.drawCircle(f16, f9, (f14 + f15) - this.mShadeRadius, this.mPaintShader);
                }
            }
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.mWidth = View.MeasureSpec.getSize(i);
        int size = View.MeasureSpec.getSize(i2);
        this.mHeight = size;
        setMeasuredDimension(this.mWidth, size);
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        float f = this.mRadiusBig;
        float f2 = this.mRadiusSpace;
        this.mFirstItemRange = f + f2 + (this.mItemWidth / 2.0f);
        this.mBigRadiusWidth = (f + f2) * 2.0f;
        int action = motionEvent.getAction();
        int i = 0;
        if (action == 0) {
            float x = motionEvent.getX();
            this.mDivX = x;
            this.mLastX = x;
            float y = motionEvent.getY();
            this.mDivY = y;
            this.mLastY = y;
            this.mClickPosition = -1;
            if (motionEvent.getX() < this.mBigRadiusWidth) {
                this.mClickPosition = 0;
            } else if (motionEvent.getX() >= this.mWidth - this.mBigRadiusWidth) {
                this.mClickPosition = this.mItemCount - 1;
            } else if (motionEvent.getX() >= this.mFirstItemRange) {
                float x2 = motionEvent.getX() - this.mFirstItemRange;
                float f3 = this.mItemWidth;
                int i2 = ((int) (x2 / f3)) + 1;
                this.mClickPosition = i2;
                if (Math.abs(((this.mBigRadiusWidth / 2.0f) + (f3 * i2)) - motionEvent.getX()) > this.mBigRadiusWidth / 2.0f) {
                    this.mClickPosition = -1;
                }
            }
            if (this.mClickPosition == this.mCurPosition) {
                this.mCanDrag = true;
                this.mIsClickValid = false;
                float f4 = this.mDivX;
                float f5 = this.mBigRadiusWidth;
                if (f4 < f5 / 2.0f) {
                    f4 = f5 / 2.0f;
                    this.mDivX = f4;
                }
                int i3 = this.mWidth;
                if (f4 > i3 - (f5 / 2.0f)) {
                    this.mDivX = i3 - (f5 / 2.0f);
                }
                invalidate();
            } else {
                this.mCanDrag = false;
                this.mIsClickValid = true;
            }
            return true;
        }
        if (action == 1) {
            this.mCanDrag = false;
            if (!this.mIsClickValid) {
                invalidate();
                OnProgressChangeListener onProgressChangeListener = this.mListener;
                if (onProgressChangeListener != null) {
                    onProgressChangeListener.onClick(this.mCurPosition);
                }
                return true;
            }
            if (Math.abs(motionEvent.getX() - this.mLastX) < this.mTouchSlop && Math.abs(motionEvent.getY() - this.mLastY) < this.mTouchSlop) {
                if (motionEvent.getX() >= this.mWidth - this.mFirstItemRange) {
                    i = this.mItemCount - 1;
                } else if (motionEvent.getX() >= this.mFirstItemRange) {
                    i = ((int) ((motionEvent.getX() - this.mFirstItemRange) / this.mItemWidth)) + 1;
                }
                if (this.mCurPosition != i) {
                    this.mCurPosition = i;
                }
                invalidate();
                OnProgressChangeListener onProgressChangeListener2 = this.mListener;
                if (onProgressChangeListener2 != null) {
                    onProgressChangeListener2.onClick(this.mCurPosition);
                }
            }
            return true;
        }
        if (action != 2) {
            return super.onTouchEvent(motionEvent);
        }
        if (!this.mCanDrag) {
            return true;
        }
        getParent().requestDisallowInterceptTouchEvent(true);
        this.mDivX = motionEvent.getX();
        this.mDivY = motionEvent.getY();
        float f6 = this.mDivX;
        float f7 = this.mBigRadiusWidth;
        if (f6 < f7 / 2.0f) {
            f6 = f7 / 2.0f;
            this.mDivX = f6;
        }
        int i4 = this.mWidth;
        if (f6 > i4 - (f7 / 2.0f)) {
            this.mDivX = i4 - (f7 / 2.0f);
        }
        this.mIsClickValid = false;
        if (motionEvent.getX() >= this.mWidth - this.mFirstItemRange) {
            i = this.mItemCount - 1;
        } else if (motionEvent.getX() >= this.mFirstItemRange) {
            i = ((int) ((motionEvent.getX() - this.mFirstItemRange) / this.mItemWidth)) + 1;
        }
        if (this.mCurPosition != i) {
            this.mCurPosition = i;
        }
        invalidate();
        return true;
    }

    public SettingProgressView setCurPosition(int i) {
        if (i < 0) {
            i = 0;
        }
        this.mCurPosition = i;
        return this;
    }

    public void setOnProgressChangeListener(OnProgressChangeListener onProgressChangeListener) {
        this.mListener = onProgressChangeListener;
    }

    public SettingProgressView setmItemCount(int i) {
        this.mItemCount = i;
        return this;
    }
}
