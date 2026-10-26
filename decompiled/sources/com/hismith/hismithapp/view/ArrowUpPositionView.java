package com.hismith.hismithapp.view;

import android.animation.ValueAnimator;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Shader;
import android.util.AttributeSet;
import android.view.View;
import android.view.animation.LinearInterpolator;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class ArrowUpPositionView extends View {
    Bitmap bitmap;
    private int[] color;
    Rect dstRect;
    private float fAnimY;
    int height;
    int mBitHeight;
    int mBitWidth;
    private float mDy;
    private LinearGradient mGradient;
    private Matrix mMatrix;
    Paint mPaint;
    RectF mRect;
    private float[] positions;
    Rect srcRect;
    int width;
    PorterDuffXfermode xfermode;

    public ArrowUpPositionView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.color = new int[]{Color.parseColor("#ffffff"), Color.parseColor("#333333"), Color.parseColor("#ffffff"), Color.parseColor("#333333")};
        this.positions = new float[]{0.0f, 0.33f, 0.66f, 1.0f};
        initType(attributeSet);
        initView();
    }

    private void initType(AttributeSet attributeSet) {
        TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, R.styleable.ArrowUpPositionView);
        int color = obtainStyledAttributes.getColor(R.styleable.ArrowUpPositionView_highColor, Color.parseColor("#ffffff"));
        int color2 = obtainStyledAttributes.getColor(R.styleable.ArrowUpPositionView_lowColor, Color.parseColor("#333333"));
        obtainStyledAttributes.recycle();
        this.color = new int[]{color, color2, color, color2};
    }

    private void initView() {
        Paint paint = new Paint();
        this.mPaint = paint;
        paint.setAntiAlias(true);
        this.mPaint.setColor(ViewCompat.MEASURED_STATE_MASK);
        this.xfermode = new PorterDuffXfermode(PorterDuff.Mode.SRC_IN);
        Bitmap decodeResource = BitmapFactory.decodeResource(getResources(), R.mipmap.narrow);
        this.bitmap = decodeResource;
        this.mBitWidth = decodeResource.getWidth();
        this.mBitHeight = this.bitmap.getHeight();
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int saveLayer = canvas.saveLayer(0.0f, 0.0f, this.width, this.height, this.mPaint, 31);
        canvas.drawBitmap(this.bitmap, this.srcRect, this.dstRect, this.mPaint);
        this.mPaint.setXfermode(this.xfermode);
        Matrix matrix = this.mMatrix;
        if (matrix == null) {
            matrix = new Matrix();
            this.mMatrix = matrix;
        }
        matrix.setTranslate(0.0f, this.mDy);
        this.mGradient.setLocalMatrix(this.mMatrix);
        this.mPaint.setShader(this.mGradient);
        canvas.drawRect(0.0f, 0.0f, this.width, this.height, this.mPaint);
        this.mPaint.setXfermode(null);
        canvas.restoreToCount(saveLayer);
    }

    @Override // android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        ValueAnimator ofFloat = ValueAnimator.ofFloat(0.0f, -(this.height * 2));
        ofFloat.setInterpolator(new LinearInterpolator());
        ofFloat.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.ArrowUpPositionView.1
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                ArrowUpPositionView.this.mDy = ((Float) valueAnimator.getAnimatedValue()).floatValue();
                ArrowUpPositionView.this.postInvalidate();
            }
        });
        ofFloat.setRepeatMode(1);
        ofFloat.setRepeatCount(-1);
        ofFloat.setDuration(1000L);
        ofFloat.start();
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        this.width = i;
        this.height = i2;
        this.srcRect = new Rect(0, 0, this.mBitWidth, this.mBitHeight);
        this.dstRect = new Rect(0, 0, this.width, this.height);
        this.fAnimY = this.height * 3;
        LinearGradient linearGradient = this.mGradient;
        if (linearGradient == null) {
            LinearGradient linearGradient2 = new LinearGradient(0.0f, 0.0f, 0.0f, this.fAnimY, this.color, this.positions, Shader.TileMode.CLAMP);
            this.mGradient = linearGradient2;
            linearGradient = linearGradient2;
        }
        this.mPaint.setShader(linearGradient);
    }
}
