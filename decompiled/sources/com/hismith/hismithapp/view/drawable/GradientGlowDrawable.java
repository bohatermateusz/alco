package com.hismith.hismithapp.view.drawable;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorFilter;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.drawable.Drawable;
import com.facebook.internal.FacebookRequestErrorClassification;

/* loaded from: classes3.dex */
public class GradientGlowDrawable extends Drawable {
    private final float cornerRadius;
    private final Paint fillPaint;
    private final Paint glowPaint;
    private final float glowRadius;
    private final Paint strokePaint;
    private final float strokeWidth;

    public GradientGlowDrawable(float f, float f2, float f3) {
        Paint paint = new Paint(1);
        this.fillPaint = paint;
        Paint paint2 = new Paint(1);
        this.glowPaint = paint2;
        Paint paint3 = new Paint(1);
        this.strokePaint = paint3;
        this.cornerRadius = f;
        this.glowRadius = f2;
        this.strokeWidth = f3;
        paint.setStyle(Paint.Style.FILL);
        paint2.setStyle(Paint.Style.FILL);
        paint3.setStyle(Paint.Style.STROKE);
        paint3.setStrokeWidth(f3);
    }

    @Override // android.graphics.drawable.Drawable
    public void draw(Canvas canvas) {
        RectF rectF = new RectF(getBounds());
        float f = this.glowRadius;
        rectF.inset(f, f);
        float min = Math.min(this.cornerRadius, rectF.height() / 2.0f);
        this.glowPaint.setShader(null);
        this.glowPaint.setColor(Color.rgb(113, 42, 255));
        this.glowPaint.setShadowLayer(this.glowRadius, 0.0f, 0.0f, Color.argb(FacebookRequestErrorClassification.EC_INVALID_TOKEN, 105, 35, 255));
        canvas.drawRoundRect(rectF, min, min, this.glowPaint);
        this.glowPaint.clearShadowLayer();
        this.fillPaint.setShader(new LinearGradient(rectF.left, 0.0f, rectF.right, 0.0f, new int[]{Color.rgb(199, 65, 255), Color.rgb(117, 43, 255), Color.rgb(62, 31, 238)}, new float[]{0.0f, 0.48f, 1.0f}, Shader.TileMode.CLAMP));
        canvas.drawRoundRect(rectF, min, min, this.fillPaint);
        if (this.strokeWidth > 0.0f) {
            this.strokePaint.setShader(new LinearGradient(rectF.left, 0.0f, rectF.right, 0.0f, new int[]{-1, Color.rgb(223, 175, 255), Color.rgb(183, 139, 255)}, (float[]) null, Shader.TileMode.CLAMP));
            canvas.drawRoundRect(rectF, min, min, this.strokePaint);
        }
    }

    @Override // android.graphics.drawable.Drawable
    public int getOpacity() {
        return -3;
    }

    @Override // android.graphics.drawable.Drawable
    public void setAlpha(int i) {
        this.fillPaint.setAlpha(i);
        this.glowPaint.setAlpha(i);
        this.strokePaint.setAlpha(i);
    }

    @Override // android.graphics.drawable.Drawable
    public void setColorFilter(ColorFilter colorFilter) {
        this.fillPaint.setColorFilter(colorFilter);
        this.glowPaint.setColorFilter(colorFilter);
        this.strokePaint.setColorFilter(colorFilter);
    }
}
