package com.hismith.hismithapp.view;

import android.animation.ValueAnimator;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Shader;
import android.graphics.drawable.BitmapDrawable;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.LinearInterpolator;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class BothWayArrowView extends View {
    private int bitmapHeight;
    private int bitmapWidth;
    private int[] color;
    private int fAnimY;
    private int height;
    private Bitmap mBitmap;
    private float mDy;
    private float mEndY;
    private LinearGradient mGradient;
    private Paint mGradientPaint;
    private Matrix mMatrix;
    private OnValueListener onValueListener;
    private float[] positions;
    private int width;

    public interface OnValueListener {
        void onValueChange(float f);
    }

    public BothWayArrowView(Context context) {
        this(context, null);
    }

    public BothWayArrowView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public BothWayArrowView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.color = new int[]{Color.parseColor("#32333333"), Color.parseColor("#FFFFFF"), Color.parseColor("#32333333")};
        this.positions = new float[]{0.0f, 0.5f, 1.0f};
        initType(attributeSet);
        init();
    }

    private void init() {
        Bitmap bitmap = ((BitmapDrawable) ContextCompat.getDrawable(getContext(), R.mipmap.ic_touch_gesture_arrow)).getBitmap();
        this.mBitmap = bitmap;
        this.bitmapWidth = bitmap.getWidth();
        this.bitmapHeight = this.mBitmap.getHeight();
        Paint paint = new Paint();
        this.mGradientPaint = paint;
        paint.setAntiAlias(true);
    }

    private void initType(AttributeSet attributeSet) {
        TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, R.styleable.BothWayArrowView);
        int color = obtainStyledAttributes.getColor(R.styleable.BothWayArrowView_highColor, Color.parseColor("#FFFFFF"));
        int color2 = obtainStyledAttributes.getColor(R.styleable.BothWayArrowView_lowColor, Color.parseColor("#32333333"));
        obtainStyledAttributes.recycle();
        this.color = new int[]{color2, color, color2};
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Matrix matrix = this.mMatrix;
        if (matrix == null) {
            matrix = new Matrix();
            this.mMatrix = matrix;
        }
        matrix.setTranslate(0.0f, this.mDy);
        this.mGradient.setLocalMatrix(this.mMatrix);
        this.mGradientPaint.setShader(this.mGradient);
        canvas.drawBitmap(this.mBitmap.extractAlpha(), 0.0f, 0.0f, this.mGradientPaint);
    }

    @Override // android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        ValueAnimator ofFloat = ValueAnimator.ofFloat(0.0f, this.mEndY);
        ofFloat.setInterpolator(new LinearInterpolator());
        ofFloat.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.BothWayArrowView.1
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                BothWayArrowView.this.mDy = ((Float) valueAnimator.getAnimatedValue()).floatValue();
                if (BothWayArrowView.this.onValueListener != null) {
                    BothWayArrowView.this.onValueListener.onValueChange(BothWayArrowView.this.mDy / BothWayArrowView.this.mEndY);
                }
                BothWayArrowView.this.postInvalidate();
            }
        });
        ofFloat.setRepeatMode(2);
        ofFloat.setRepeatCount(-1);
        ofFloat.setDuration(1000L);
        ofFloat.start();
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        this.width = View.MeasureSpec.getSize(i);
        int size = View.MeasureSpec.getSize(i2);
        this.height = size;
        int i3 = this.bitmapHeight;
        if (size < i3) {
            this.bitmapWidth = (int) (((size * 1.0f) / i3) * this.bitmapWidth);
            this.bitmapHeight = size;
            LogUtil.e("height:" + this.height + ", bitmapHeight:" + this.bitmapHeight + ", bitmapWidth:" + this.bitmapWidth);
        }
        int i4 = this.bitmapWidth;
        this.width = i4;
        Bitmap createScaledBitmap = Bitmap.createScaledBitmap(this.mBitmap, i4, this.bitmapHeight, true);
        this.mBitmap = createScaledBitmap;
        createScaledBitmap.setConfig(Bitmap.Config.ARGB_8888);
        int i5 = this.height;
        this.mEndY = i5 * 2;
        this.fAnimY = i5 * 3;
        LinearGradient linearGradient = this.mGradient;
        if (linearGradient == null) {
            LinearGradient linearGradient2 = new LinearGradient(0.0f, r9 - this.fAnimY, 0.0f, this.height, this.color, this.positions, Shader.TileMode.CLAMP);
            this.mGradient = linearGradient2;
            linearGradient = linearGradient2;
        }
        this.mGradientPaint.setShader(linearGradient);
        setMeasuredDimension(this.width, this.height);
    }

    public void setOnValueListener(OnValueListener onValueListener) {
        this.onValueListener = onValueListener;
    }
}
