package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.FrameLayout;
import android.widget.TextView;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class HoleMaskView extends FrameLayout {
    private boolean hasManualRect;
    private float holePaddingDp;
    private final Paint holePaint;
    private float holeRadiusDp;
    private final RectF holeRect;
    private final RectF manualRect;
    private final Paint maskPaint;
    private final Paint ringPaint;
    private final int[] selfLocation;
    private final View.OnLayoutChangeListener targetLayoutListener;
    private final int[] targetLocation;
    private View targetView;
    private boolean tipAbove;
    private View tipArrow;
    private OnTipDismissedListener tipDismissedListener;
    private View tipView;

    public interface OnTipDismissedListener {
        void onTipDismissed(boolean z);
    }

    public HoleMaskView(Context context) {
        this(context, null);
    }

    public HoleMaskView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        Paint paint = new Paint();
        this.maskPaint = paint;
        Paint paint2 = new Paint();
        this.holePaint = paint2;
        Paint paint3 = new Paint();
        this.ringPaint = paint3;
        this.holeRect = new RectF();
        this.targetLocation = new int[2];
        this.selfLocation = new int[2];
        this.tipAbove = false;
        this.holeRadiusDp = 0.0f;
        this.holePaddingDp = 8.0f;
        this.hasManualRect = false;
        this.manualRect = new RectF();
        this.targetLayoutListener = new View.OnLayoutChangeListener() { // from class: com.hismith.hismithapp.view.HoleMaskView$$ExternalSyntheticLambda0
            @Override // android.view.View.OnLayoutChangeListener
            public final void onLayoutChange(View view, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
                HoleMaskView.this.m630lambda$new$0$comhismithhismithappviewHoleMaskView(view, i, i2, i3, i4, i5, i6, i7, i8);
            }
        };
        setClickable(true);
        setFocusable(true);
        setWillNotDraw(false);
        paint.setColor(-1728053248);
        paint.setAntiAlias(true);
        paint2.setColor(ViewCompat.MEASURED_STATE_MASK);
        paint2.setAntiAlias(true);
        paint2.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        paint3.setStyle(Paint.Style.STROKE);
        paint3.setStrokeWidth(dp(1.5f));
        paint3.setColor(-65302);
        paint3.setAntiAlias(true);
        setLayerType(1, null);
        LayoutInflater.from(context).inflate(R.layout.view_hole_mask_tip, (ViewGroup) this, true);
        this.tipView = findViewById(R.id.layout_hole_mask_tip_content);
        this.tipArrow = findViewById(R.id.hole_mask_tip_arrow);
        findViewById(R.id.btn_hole_mask_tip_got_it).setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.view.HoleMaskView$$ExternalSyntheticLambda1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                HoleMaskView.this.m631lambda$new$1$comhismithhismithappviewHoleMaskView(view);
            }
        });
    }

    private float dp(float f) {
        return f * getResources().getDisplayMetrics().density;
    }

    private void flipTipArrow(boolean z, float f, int i, int i2) {
        View view = this.tipArrow;
        if (view == null) {
            return;
        }
        if (this.tipAbove != z) {
            this.tipAbove = z;
            if (z) {
                View view2 = this.tipView;
                int measuredHeight = view2 != null ? view2.getMeasuredHeight() : 0;
                this.tipArrow.setTranslationY(measuredHeight > 0 ? measuredHeight - dp(6.0f) : 0.0f);
            } else {
                view.setTranslationY(0.0f);
            }
        }
        float f2 = i - f;
        float f3 = i2;
        this.tipArrow.setTranslationX(Math.max(dp(6.0f), Math.min(f2, f3 - dp(6.0f))) - (f3 - dp(23.0f)));
    }

    private void updateAll() {
        updateHole();
        updateTipPosition();
    }

    private void updateHole() {
        if (this.hasManualRect) {
            this.holeRect.set(this.manualRect);
            invalidate();
            return;
        }
        if (this.targetView == null || !isAttachedToWindow() || this.targetView.getVisibility() != 0 || this.targetView.getWidth() <= 0 || this.targetView.getHeight() <= 0) {
            return;
        }
        this.targetView.getLocationInWindow(this.targetLocation);
        getLocationInWindow(this.selfLocation);
        float dp = dp(this.holePaddingDp);
        int i = this.targetLocation[0];
        int[] iArr = this.selfLocation;
        float f = (i - iArr[0]) - dp;
        float f2 = (r1[1] - iArr[1]) - dp;
        float f3 = dp * 2.0f;
        this.holeRect.set(f, f2, this.targetView.getWidth() + f + f3, this.targetView.getHeight() + f2 + f3);
        invalidate();
    }

    private void updateTipPosition() {
        if (this.tipView == null || this.targetView == null || !isAttachedToWindow() || this.targetView.getVisibility() != 0 || this.targetView.getWidth() <= 0 || this.targetView.getHeight() <= 0) {
            return;
        }
        int width = getWidth();
        int height = getHeight();
        if (width <= 0 || height <= 0) {
            return;
        }
        this.targetView.getLocationInWindow(this.targetLocation);
        getLocationInWindow(this.selfLocation);
        this.tipView.measure(View.MeasureSpec.makeMeasureSpec(0, 0), View.MeasureSpec.makeMeasureSpec(0, 0));
        int measuredWidth = this.tipView.getMeasuredWidth();
        int measuredHeight = this.tipView.getMeasuredHeight();
        if (measuredWidth <= 0 || measuredHeight <= 0) {
            return;
        }
        int dp = (int) dp(10.0f);
        int[] iArr = this.targetLocation;
        int i = iArr[0];
        int[] iArr2 = this.selfLocation;
        int i2 = i - iArr2[0];
        boolean z = true;
        int i3 = iArr[1] - iArr2[1];
        int width2 = this.targetView.getWidth() + i2;
        int height2 = this.targetView.getHeight() + i3;
        int width3 = (this.targetView.getWidth() / 2) + i2;
        ViewGroup.MarginLayoutParams marginLayoutParams = (ViewGroup.MarginLayoutParams) this.tipView.getLayoutParams();
        if (marginLayoutParams == null) {
            return;
        }
        int i4 = height2 + dp;
        if (i4 + measuredHeight <= height) {
            marginLayoutParams.topMargin = i4;
            z = false;
        } else {
            marginLayoutParams.topMargin = Math.max(dp, (i3 - dp) - measuredHeight);
        }
        float max = Math.max(dp, Math.min(i2 <= dp ? i2 : width2 >= width - dp ? width2 - measuredWidth : width3 - (measuredWidth / 2.0f), (width - measuredWidth) - dp));
        marginLayoutParams.leftMargin = (int) max;
        marginLayoutParams.rightMargin = 0;
        this.tipView.setLayoutParams(marginLayoutParams);
        flipTipArrow(z, max, width3, measuredWidth);
    }

    /* renamed from: lambda$new$0$com-hismith-hismithapp-view-HoleMaskView, reason: not valid java name */
    /* synthetic */ void m630lambda$new$0$comhismithhismithappviewHoleMaskView(View view, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
        updateAll();
    }

    /* renamed from: lambda$new$1$com-hismith-hismithapp-view-HoleMaskView, reason: not valid java name */
    /* synthetic */ void m631lambda$new$1$comhismithhismithappviewHoleMaskView(View view) {
        CheckBox checkBox = (CheckBox) findViewById(R.id.chk_hole_mask_tip_no_more);
        OnTipDismissedListener onTipDismissedListener = this.tipDismissedListener;
        if (onTipDismissedListener != null) {
            onTipDismissedListener.onTipDismissed(checkBox.isChecked());
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        View view = this.targetView;
        if (view != null) {
            view.removeOnLayoutChangeListener(this.targetLayoutListener);
            this.targetView = null;
        }
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        canvas.drawRect(0.0f, 0.0f, getWidth(), getHeight(), this.maskPaint);
        if (this.holeRect.isEmpty()) {
            return;
        }
        float f = this.holeRadiusDp;
        if (f <= 0.0f) {
            f = 8.0f;
        }
        float dp = dp(f);
        canvas.drawRoundRect(this.holeRect, dp, dp, this.holePaint);
        canvas.drawRoundRect(this.holeRect, dp, dp, this.ringPaint);
    }

    @Override // android.widget.FrameLayout, android.view.ViewGroup, android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        updateAll();
    }

    public void setGotItText(CharSequence charSequence) {
        ((TextView) findViewById(R.id.btn_hole_mask_tip_got_it)).setText(charSequence);
    }

    public void setHolePadding(float f) {
        this.holePaddingDp = f;
        updateAll();
    }

    public void setHoleRadius(float f) {
        this.holeRadiusDp = f;
        invalidate();
    }

    public void setHoleRect(RectF rectF) {
        View view = this.targetView;
        if (view != null) {
            view.removeOnLayoutChangeListener(this.targetLayoutListener);
            this.targetView = null;
        }
        this.hasManualRect = true;
        this.manualRect.set(rectF);
        updateAll();
    }

    public void setNoMoreText(CharSequence charSequence) {
        ((TextView) findViewById(R.id.chk_hole_mask_tip_no_more)).setText(charSequence);
    }

    public void setOnTipDismissedListener(OnTipDismissedListener onTipDismissedListener) {
        this.tipDismissedListener = onTipDismissedListener;
    }

    public void setTargetView(View view) {
        View view2 = this.targetView;
        if (view2 == view) {
            return;
        }
        if (view2 != null) {
            view2.removeOnLayoutChangeListener(this.targetLayoutListener);
        }
        this.targetView = view;
        this.hasManualRect = false;
        if (view != null) {
            view.addOnLayoutChangeListener(this.targetLayoutListener);
        }
        updateAll();
    }

    public void setTipText(CharSequence charSequence) {
        ((TextView) findViewById(R.id.txt_hole_mask_tip)).setText(charSequence);
    }
}
