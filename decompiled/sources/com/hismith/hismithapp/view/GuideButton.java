package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Shader;
import android.util.AttributeSet;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.media3.extractor.ts.TsExtractor;
import com.facebook.internal.FacebookRequestErrorClassification;
import com.hismith.hismithapp.fragment.PersonalFragment;

/* loaded from: classes3.dex */
public class GuideButton extends AppCompatTextView {
    private final Paint borderPaint;
    private final Paint starPaint;

    public GuideButton(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.borderPaint = new Paint(1);
        Paint paint = new Paint(1);
        this.starPaint = paint;
        setLayerType(1, null);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.rgb(FacebookRequestErrorClassification.EC_INVALID_TOKEN, 126, 255));
    }

    private float dp(float f) {
        return f * getResources().getDisplayMetrics().density;
    }

    private void drawStar(Canvas canvas, float f, float f2, float f3, float f4) {
        Path path = new Path();
        for (int i = 0; i < 8; i++) {
            double d = ((i * 3.141592653589793d) / 4.0d) - 1.5707963267948966d;
            float f5 = i % 2 == 0 ? f3 : f4;
            float cos = (((float) Math.cos(d)) * f5) + f;
            float sin = (((float) Math.sin(d)) * f5) + f2;
            if (i == 0) {
                path.moveTo(cos, sin);
            } else {
                path.lineTo(cos, sin);
            }
        }
        path.close();
        canvas.drawPath(path, this.starPaint);
    }

    @Override // android.widget.TextView, android.view.View
    protected void onDraw(Canvas canvas) {
        float dp = dp(2.0f);
        float height = getHeight() / 2.0f;
        Paint paint = new Paint(1);
        paint.setColor(Color.rgb(9, 5, 39));
        canvas.drawRoundRect(dp, dp, getWidth() - dp, getHeight() - dp, height, height, paint);
        this.borderPaint.setStyle(Paint.Style.STROKE);
        this.borderPaint.setStrokeWidth(dp);
        this.borderPaint.setShader(new LinearGradient(0.0f, 0.0f, getWidth(), 0.0f, new int[]{Color.rgb(221, PersonalFragment.requestInfo, 255), Color.rgb(128, 47, 255), Color.rgb(67, 19, 209)}, (float[]) null, Shader.TileMode.CLAMP));
        this.borderPaint.setShadowLayer(dp(5.0f), 0.0f, 0.0f, Color.argb(TsExtractor.TS_STREAM_TYPE_HDMV_DTS, TsExtractor.TS_STREAM_TYPE_HDMV_DTS, 50, 255));
        canvas.drawRoundRect(dp, dp, getWidth() - dp, getHeight() - dp, height, height, this.borderPaint);
        super.onDraw(canvas);
    }
}
