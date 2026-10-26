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
import android.graphics.PointF;
import android.graphics.Shader;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.LinearInterpolator;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.MeasureUtil;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class ArrowView2 extends View {
    private int[] color;
    private int direction;
    private float fAnimY;
    private float height;
    private int mArrowCount;
    private List<PointF[]> mArrowList;
    private Paint mArrowPaint;
    private Path mArrowPath;
    private float mArrowStroke;
    private float mDy;
    private LinearGradient mGradient;
    private Paint mGradientPaint;
    private Matrix mMatrix;
    private float[] positions;
    private float width;

    public ArrowView2(Context context) {
        this(context, null);
    }

    public ArrowView2(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public ArrowView2(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.color = new int[]{Color.parseColor("#333333"), Color.parseColor("#ffffff"), Color.parseColor("#333333"), Color.parseColor("#ffffff")};
        this.positions = new float[]{0.0f, 0.33f, 0.66f, 1.0f};
        this.mArrowCount = 3;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.ArrowView2);
        this.direction = obtainStyledAttributes.getInt(R.styleable.ArrowView2_direction, 0);
        this.mArrowCount = obtainStyledAttributes.getInt(R.styleable.ArrowView2_arrowCount, 0);
        this.mArrowStroke = obtainStyledAttributes.getDimension(R.styleable.ArrowView2_arrowStroke, MeasureUtil.dip2px(getContext(), 3.0f));
        int color = obtainStyledAttributes.getColor(R.styleable.ArrowView2_highColor, Color.parseColor("#ffffff"));
        int color2 = obtainStyledAttributes.getColor(R.styleable.ArrowView2_lowColor, Color.parseColor("#333333"));
        obtainStyledAttributes.recycle();
        this.color = new int[]{color, color2, color, color2};
        init();
    }

    private void init() {
        this.mArrowStroke = MeasureUtil.dip2px(getContext(), 3.0f);
        this.mArrowList = new ArrayList();
        Paint paint = new Paint();
        this.mGradientPaint = paint;
        paint.setAntiAlias(true);
        Paint paint2 = new Paint();
        this.mArrowPaint = paint2;
        paint2.setAntiAlias(true);
        this.mArrowPaint.setStrokeWidth(this.mArrowStroke);
        this.mArrowPaint.setStyle(Paint.Style.STROKE);
        this.mArrowPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mArrowPaint.setStrokeCap(Paint.Cap.ROUND);
        this.mArrowPath = new Path();
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
        int i = 0;
        while (true) {
            int size = this.mArrowList.size();
            Path path = this.mArrowPath;
            if (i >= size) {
                canvas.drawPath(path, this.mArrowPaint);
                return;
            }
            path.moveTo(this.mArrowList.get(i)[0].x, this.mArrowList.get(i)[0].y);
            this.mArrowPath.lineTo(this.mArrowList.get(i)[1].x, this.mArrowList.get(i)[1].y);
            this.mArrowPath.lineTo(this.mArrowList.get(i)[2].x, this.mArrowList.get(i)[2].y);
            i++;
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        float size = View.MeasureSpec.getSize(i2);
        this.height = size;
        this.width = size * 0.6f;
        int mode = View.MeasureSpec.getMode(i);
        int size2 = View.MeasureSpec.getSize(i);
        if (mode == 1073741824) {
            this.width = size2;
        }
        int i3 = this.direction;
        if (i3 == 0) {
            this.fAnimY = this.height * 3.0f;
            if (this.mGradient == null) {
                this.mGradient = new LinearGradient(0.0f, 0.0f, 0.0f, this.fAnimY, this.color, this.positions, Shader.TileMode.CLAMP);
            }
        } else if (i3 == 1) {
            this.fAnimY = -(this.height * 2.0f);
            if (this.mGradient == null) {
                this.mGradient = new LinearGradient(0.0f, this.fAnimY, 0.0f, this.height, this.color, this.positions, Shader.TileMode.CLAMP);
            }
        }
        this.mGradientPaint.setShader(this.mGradient);
        this.mArrowPaint.setShader(this.mGradient);
        setMeasuredDimension((int) this.width, (int) this.height);
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        int i5;
        ValueAnimator valueAnimator;
        super.onSizeChanged(i, i2, i3, i4);
        this.mArrowList.clear();
        float f = this.height / this.mArrowCount;
        int i6 = 0;
        while (true) {
            int i7 = this.mArrowCount;
            i5 = this.direction;
            valueAnimator = null;
            PointF[] pointFArr = null;
            if (i6 >= i7) {
                break;
            }
            if (i5 == 0) {
                float f2 = (i6 + 1) * f;
                pointFArr = new PointF[]{new PointF(this.mArrowStroke / 2.0f, f2), new PointF(this.width / 2.0f, (i6 * f) + (this.mArrowStroke / 2.0f)), new PointF(this.width - (this.mArrowStroke / 2.0f), f2)};
            } else if (i5 == 1) {
                float f3 = i6 * f;
                pointFArr = new PointF[]{new PointF(this.mArrowStroke / 2.0f, f3), new PointF(this.width / 2.0f, ((i6 + 1) * f) - (this.mArrowStroke / 2.0f)), new PointF(this.width - (this.mArrowStroke / 2.0f), f3)};
            }
            if (pointFArr != null) {
                this.mArrowList.add(pointFArr);
            }
            i6++;
        }
        if (i5 == 0) {
            valueAnimator = ValueAnimator.ofFloat(0.0f, -(this.height * 2.0f));
        } else if (i5 == 1) {
            valueAnimator = ValueAnimator.ofFloat(0.0f, this.height * 2.0f);
        }
        if (valueAnimator != null) {
            valueAnimator.setInterpolator(new LinearInterpolator());
            valueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.ArrowView2.1
                @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                    ArrowView2.this.mDy = ((Float) valueAnimator2.getAnimatedValue()).floatValue();
                    ArrowView2.this.postInvalidate();
                }
            });
            valueAnimator.setRepeatMode(1);
            valueAnimator.setRepeatCount(-1);
            valueAnimator.setDuration(1000L);
            valueAnimator.start();
        }
    }
}
