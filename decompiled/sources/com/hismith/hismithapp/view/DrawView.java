package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import androidx.core.internal.view.SupportMenu;

/* loaded from: classes3.dex */
public class DrawView extends View {
    private Paint paint;

    public DrawView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, 0);
        init();
    }

    public DrawView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        init();
    }

    private void init() {
        Paint paint = new Paint(4);
        this.paint = paint;
        paint.setStyle(Paint.Style.STROKE);
        this.paint.setStrokeWidth(5.0f);
        this.paint.setColor(SupportMenu.CATEGORY_MASK);
        this.paint.setAntiAlias(true);
        this.paint.setDither(true);
        this.paint.setStrokeJoin(Paint.Join.ROUND);
        this.paint.setStrokeCap(Paint.Cap.ROUND);
    }
}
