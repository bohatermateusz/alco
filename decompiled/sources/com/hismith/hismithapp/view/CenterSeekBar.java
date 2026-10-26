package com.hismith.hismithapp.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.util.AttributeSet;
import androidx.appcompat.widget.AppCompatSeekBar;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class CenterSeekBar extends AppCompatSeekBar {
    private int backgroundColor;
    private Paint paint;
    private int progressColor;
    private Rect rect;
    private float seekbar_height;
    private int thumbColor;

    public static class Builder {
        int backgroundColor;
        Context context;
        int progressColor;
        float seekbar_height;
        int thumbColor;

        public CenterSeekBar build() {
            return new CenterSeekBar(this.context, this);
        }

        public Builder setBackgroundColor(int i) {
            this.backgroundColor = i;
            return this;
        }

        public Builder setHeight(float f) {
            this.seekbar_height = f;
            return this;
        }

        public Builder setProgressColor(int i) {
            this.progressColor = i;
            return this;
        }

        public Builder setThumbColor(int i) {
            this.thumbColor = i;
            return this;
        }

        public Builder with(Context context) {
            this.context = context;
            return this;
        }
    }

    public CenterSeekBar(Context context) {
        super(context);
    }

    public CenterSeekBar(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        initAttrs(context, attributeSet);
    }

    public CenterSeekBar(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
    }

    public CenterSeekBar(Context context, Builder builder) {
        super(context);
        this.rect = new Rect();
        this.paint = new Paint();
        this.seekbar_height = builder.seekbar_height;
        this.thumbColor = ContextCompat.getColor(context, builder.thumbColor);
        this.backgroundColor = ContextCompat.getColor(context, builder.backgroundColor);
        this.progressColor = ContextCompat.getColor(context, builder.progressColor);
        getProgressDrawable().setAlpha(0);
        getThumb().setColorFilter(this.thumbColor, PorterDuff.Mode.SRC_ATOP);
    }

    public void initAttrs(Context context, AttributeSet attributeSet) {
        this.rect = new Rect();
        this.paint = new Paint();
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.CenterSeekBar);
        try {
            this.backgroundColor = obtainStyledAttributes.getColor(R.styleable.CenterSeekBar_seekbar_backgroundtint, ContextCompat.getColor(context, R.color.colorPrimaryDark));
            this.progressColor = obtainStyledAttributes.getColor(R.styleable.CenterSeekBar_seekbar_progresstint, ContextCompat.getColor(context, R.color.colorPrimary));
            this.thumbColor = obtainStyledAttributes.getColor(R.styleable.CenterSeekBar_seekbar_thumbtint, ContextCompat.getColor(context, R.color.colorPrimaryDark));
            this.seekbar_height = obtainStyledAttributes.getDimension(R.styleable.CenterSeekBar_seekbar_height, 6.0f);
            obtainStyledAttributes.recycle();
            getProgressDrawable().setAlpha(0);
            getThumb().setColorFilter(this.thumbColor, PorterDuff.Mode.SRC_ATOP);
        } catch (Throwable th) {
            obtainStyledAttributes.recycle();
            throw th;
        }
    }

    @Override // androidx.appcompat.widget.AppCompatSeekBar, android.widget.AbsSeekBar, android.widget.ProgressBar, android.view.View
    protected synchronized void onDraw(Canvas canvas) {
        int paddingLeft = getPaddingLeft() != 0 ? getPaddingLeft() : getPaddingRight();
        this.rect.set(paddingLeft, (getHeight() / 2) - (((int) this.seekbar_height) / 2), getWidth() - paddingLeft, (getHeight() / 2) + (((int) this.seekbar_height) / 2));
        this.paint.setColor(this.backgroundColor);
        canvas.drawRect(this.rect, this.paint);
        if (getProgress() > getMax() / 2) {
            this.rect.set(getWidth() / 2, (getHeight() / 2) - (((int) this.seekbar_height) / 2), (getWidth() / 2) + (((getProgress() - (getMax() / 2)) * (((getWidth() - paddingLeft) - getThumbOffset()) / 2)) / (getMax() / 2)), (getHeight() / 2) + (((int) this.seekbar_height) / 2));
            this.paint.setColor(this.progressColor);
            canvas.drawRect(this.rect, this.paint);
        }
        if (getProgress() == getMax() / 2) {
            this.rect.set(0, 0, 0, 0);
            this.paint.setColor(this.progressColor);
            canvas.drawRect(this.rect, this.paint);
        }
        if (getProgress() < getMax() / 2) {
            this.rect.set((getWidth() / 2) + (((getProgress() - (getMax() / 2)) * (((getWidth() - paddingLeft) - getThumbOffset()) / 2)) / (getMax() / 2)), (getHeight() / 2) - (((int) this.seekbar_height) / 2), getWidth() / 2, (getHeight() / 2) + (((int) this.seekbar_height) / 2));
            this.paint.setColor(this.progressColor);
            canvas.drawRect(this.rect, this.paint);
        }
        super.onDraw(canvas);
    }
}
