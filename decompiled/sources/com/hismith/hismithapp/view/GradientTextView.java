package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Shader;
import android.util.AttributeSet;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.media3.extractor.metadata.dvbsi.AppInfoTableDecoder;
import com.facebook.internal.FacebookRequestErrorClassification;

/* loaded from: classes.dex */
public class GradientTextView extends AppCompatTextView {
    public GradientTextView(Context context) {
        super(context);
        init();
    }

    public GradientTextView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        init();
    }

    public GradientTextView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        init();
    }

    private void init() {
        setTextColor(-1);
        setTypeface(getTypeface(), 1);
        setLayerType(1, null);
    }

    private void updateEffect() {
        int width = (getWidth() - getPaddingLeft()) - getPaddingRight();
        if (width <= 0) {
            return;
        }
        getPaint().setShader(new LinearGradient(0.0f, 0.0f, width, 0.0f, new int[]{Color.rgb(105, 158, 255), Color.rgb(FacebookRequestErrorClassification.EC_INVALID_TOKEN, 207, 255), -1, Color.rgb(255, 225, 255), Color.rgb(213, 137, 255)}, new float[]{0.0f, 0.22f, 0.48f, 0.7f, 1.0f}, Shader.TileMode.CLAMP));
        getPaint().setShadowLayer(12.0f, 0.0f, 0.0f, Color.argb(180, AppInfoTableDecoder.APPLICATION_INFORMATION_TABLE_ID, 76, 255));
        invalidate();
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        updateEffect();
    }
}
