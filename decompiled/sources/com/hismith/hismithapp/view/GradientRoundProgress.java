package com.hismith.hismithapp.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.SweepGradient;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.View;
import androidx.core.internal.view.SupportMenu;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class GradientRoundProgress extends View {
    private int endColor;
    private float max;
    private int midColor;
    private float numSize;
    private Paint paint;
    private float progress;
    private int progressColor;
    private float progressWidth;
    private int roundColor;
    private float roundWidth;
    private float roundWidthPercent;
    private int startAngle;
    private int startColor;
    private String text;
    private int textColor;
    private boolean textShow;
    private float textSize;
    private boolean useCustomFont;

    public GradientRoundProgress(Context context) {
        this(context, null);
    }

    public GradientRoundProgress(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public GradientRoundProgress(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.progress = 0.0f;
        this.paint = new Paint();
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.GradientRoundProgress);
        this.roundColor = obtainStyledAttributes.getColor(R.styleable.GradientRoundProgress_grp_roundColor, SupportMenu.CATEGORY_MASK);
        this.roundWidth = obtainStyledAttributes.getDimension(R.styleable.GradientRoundProgress_grp_roundWidth, 5.0f);
        this.roundWidthPercent = obtainStyledAttributes.getFloat(R.styleable.GradientRoundProgress_grp_roundWidthPercent, 0.0f);
        this.progressColor = obtainStyledAttributes.getColor(R.styleable.GradientRoundProgress_grp_progressColor, -16711936);
        this.progressWidth = obtainStyledAttributes.getDimension(R.styleable.GradientRoundProgress_grp_progressWidth, this.roundWidth);
        this.text = obtainStyledAttributes.getString(R.styleable.GradientRoundProgress_grp_text);
        this.textColor = obtainStyledAttributes.getColor(R.styleable.GradientRoundProgress_grp_textColor, -16711936);
        this.textSize = obtainStyledAttributes.getDimension(R.styleable.GradientRoundProgress_grp_textSize, 11.0f);
        this.numSize = obtainStyledAttributes.getDimension(R.styleable.GradientRoundProgress_grp_numSize, 14.0f);
        this.max = obtainStyledAttributes.getInteger(R.styleable.GradientRoundProgress_grp_max, 100);
        this.startAngle = obtainStyledAttributes.getInt(R.styleable.GradientRoundProgress_grp_startAngle, 90);
        this.textShow = obtainStyledAttributes.getBoolean(R.styleable.GradientRoundProgress_grp_textShow, true);
        this.useCustomFont = obtainStyledAttributes.getBoolean(R.styleable.GradientRoundProgress_grp_userCustomFont, false);
        this.startColor = obtainStyledAttributes.getColor(R.styleable.GradientRoundProgress_grp_startColor, -16711936);
        this.midColor = obtainStyledAttributes.getColor(R.styleable.GradientRoundProgress_grp_midColor, 0);
        this.endColor = obtainStyledAttributes.getColor(R.styleable.GradientRoundProgress_grp_endColor, -16711936);
        obtainStyledAttributes.recycle();
    }

    public synchronized float getProgress() {
        return this.progress;
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        String str;
        super.onDraw(canvas);
        int width = getWidth() / 2;
        float f = width;
        float f2 = this.roundWidth;
        int i = (int) (f - (f2 / 2.0f));
        this.paint.setStrokeWidth(f2);
        this.paint.setColor(this.roundColor);
        this.paint.setAntiAlias(true);
        this.paint.setStyle(Paint.Style.STROKE);
        canvas.drawCircle(f, f, i, this.paint);
        canvas.save();
        canvas.rotate(this.startAngle, f, f);
        this.paint.setStrokeWidth(this.progressWidth);
        this.paint.setColor(this.progressColor);
        float f3 = width - i;
        float f4 = width + i;
        RectF rectF = new RectF(f3, f3, f4, f4);
        float f5 = (this.progress * 360.0f) / this.max;
        int i2 = this.midColor;
        Paint paint = this.paint;
        if (i2 != 0) {
            paint.setShader(new SweepGradient(f, f, new int[]{this.startColor, this.midColor, this.endColor}, (float[]) null));
        } else {
            paint.setShader(new SweepGradient(f, f, new int[]{this.startColor, this.endColor}, (float[]) null));
        }
        canvas.drawArc(rectF, 0.0f, f5, false, this.paint);
        this.paint.setShader(null);
        canvas.rotate(-this.startAngle, f, f);
        canvas.restore();
        this.paint.setStrokeWidth(0.0f);
        this.paint.setColor(this.textColor);
        this.paint.setTextSize(this.textSize);
        int i3 = (int) ((this.progress / this.max) * 100.0f);
        if (!this.textShow || (str = this.text) == null || str.length() <= 0 || i3 < 0) {
            return;
        }
        this.paint.setTypeface(Typeface.DEFAULT);
        canvas.drawText(this.text, f - (this.paint.measureText(this.text) / 2.0f), this.textSize + f + 5.0f, this.paint);
        this.paint.setTextSize(this.numSize);
        canvas.drawText(i3 + "%", f - (this.paint.measureText(i3 + "%") / 2.0f), f, this.paint);
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        if (this.roundWidthPercent != 0.0f) {
            float measuredWidth = (getMeasuredWidth() / 2.0f) * this.roundWidthPercent;
            this.roundWidth = measuredWidth;
            this.progressWidth = measuredWidth;
        }
    }

    public synchronized void setMax(float f) {
        if (f < 0.0f) {
            throw new IllegalArgumentException("max not less than 0");
        }
        this.max = f;
    }

    public synchronized void setMaxAndProgress(float f, float f2) {
        this.max = f;
        if (f2 <= f) {
            f = f2;
        }
        this.progress = f;
        invalidate();
    }

    public synchronized void setProgress(float f) {
        if (f < 0.0f) {
            throw new IllegalArgumentException("progress not less than 0");
        }
        float f2 = this.max;
        if (f > f2) {
            f = f2;
        }
        this.progress = f;
        postInvalidate();
    }
}
