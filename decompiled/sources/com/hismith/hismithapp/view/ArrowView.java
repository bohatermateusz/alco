package com.hismith.hismithapp.view;

import android.animation.ValueAnimator;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Shader;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.LinearInterpolator;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class ArrowView extends View {
    private int[] color;
    private float fAnimY;
    private float height;
    private float mArrowHeight;
    private float mArrowInterval;
    private Paint mArrowPaint;
    private Path mArrowPath;
    private float mArrowStroke;
    private float mArrowWidth;
    private float mDy;
    private LinearGradient mGradient;
    private Paint mGradientPaint;
    private Matrix mMatrix;
    private float[] positions;
    private float width;

    public ArrowView(Context context) {
        this(context, null);
    }

    public ArrowView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public ArrowView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.color = new int[]{Color.parseColor("#333333"), Color.parseColor("#ffffff"), Color.parseColor("#333333"), Color.parseColor("#ffffff")};
        this.positions = new float[]{0.0f, 0.33f, 0.66f, 1.0f};
        initType(attributeSet);
        init();
    }

    private void init() {
        Paint paint = new Paint();
        this.mGradientPaint = paint;
        paint.setAntiAlias(true);
        Paint paint2 = new Paint();
        this.mArrowPaint = paint2;
        paint2.setAntiAlias(true);
        this.mArrowPaint.setStyle(Paint.Style.FILL);
        this.mArrowPath = new Path();
    }

    private void initType(AttributeSet attributeSet) {
        TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, R.styleable.ArrowView);
        int color = obtainStyledAttributes.getColor(R.styleable.ArrowView_highColor, Color.parseColor("#ffffff"));
        int color2 = obtainStyledAttributes.getColor(R.styleable.ArrowView_lowColor, Color.parseColor("#333333"));
        obtainStyledAttributes.recycle();
        this.color = new int[]{color2, color, color2, color};
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
        this.mArrowPaint.setShader(this.mGradient);
        this.mArrowPath.reset();
        this.mArrowPath.moveTo(0.0f, 0.0f);
        this.mArrowPath.lineTo(this.width / 2.0f, this.mArrowHeight);
        this.mArrowPath.lineTo(this.width, 0.0f);
        this.mArrowPath.lineTo(this.width - this.mArrowWidth, 0.0f);
        this.mArrowPath.lineTo(this.width / 2.0f, this.mArrowHeight - this.mArrowStroke);
        this.mArrowPath.lineTo(this.mArrowWidth, 0.0f);
        float f = this.mArrowInterval;
        this.mArrowPath.moveTo(0.0f, f);
        this.mArrowPath.lineTo(this.width / 2.0f, this.mArrowInterval + this.mArrowHeight);
        this.mArrowPath.lineTo(this.width, f);
        this.mArrowPath.lineTo(this.width - this.mArrowWidth, f);
        this.mArrowPath.lineTo(this.width / 2.0f, (this.mArrowInterval + this.mArrowHeight) - this.mArrowStroke);
        this.mArrowPath.lineTo(this.mArrowWidth, f);
        canvas.drawPath(this.mArrowPath, this.mArrowPaint);
    }

    @Override // android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        ValueAnimator ofFloat = ValueAnimator.ofFloat(0.0f, this.height * 2.0f);
        ofFloat.setInterpolator(new LinearInterpolator());
        ofFloat.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.ArrowView.1
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                ArrowView.this.mDy = ((Float) valueAnimator.getAnimatedValue()).floatValue();
                ArrowView.this.postInvalidate();
            }
        });
        ofFloat.setRepeatMode(1);
        ofFloat.setRepeatCount(-1);
        ofFloat.setDuration(1000L);
        ofFloat.start();
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        float size = View.MeasureSpec.getSize(i2);
        this.height = size;
        this.width = size * 2.8f;
        int mode = View.MeasureSpec.getMode(i);
        int size2 = View.MeasureSpec.getSize(i);
        if (mode == 1073741824) {
            this.width = size2;
        }
        float f = this.height;
        this.mArrowStroke = 0.133f * f;
        this.mArrowWidth = this.width * 0.071f;
        this.mArrowHeight = 0.777f * f;
        this.mArrowInterval = 0.222f * f;
        this.fAnimY = -(f * 2.0f);
        LinearGradient linearGradient = this.mGradient;
        if (linearGradient == null) {
            LinearGradient linearGradient2 = new LinearGradient(0.0f, this.fAnimY, 0.0f, this.height, this.color, this.positions, Shader.TileMode.CLAMP);
            this.mGradient = linearGradient2;
            linearGradient = linearGradient2;
        }
        this.mGradientPaint.setShader(linearGradient);
        this.mArrowPaint.setShader(this.mGradient);
        setMeasuredDimension((int) this.width, (int) this.height);
    }
}
