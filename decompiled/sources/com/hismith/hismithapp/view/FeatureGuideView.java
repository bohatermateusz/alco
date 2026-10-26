package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.Shader;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatCheckBox;
import androidx.core.location.LocationRequestCompat;
import androidx.core.view.ViewCompat;
import androidx.media3.extractor.metadata.dvbsi.AppInfoTableDecoder;
import com.facebook.internal.FacebookRequestErrorClassification;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.drawable.GradientGlowDrawable;

/* loaded from: classes3.dex */
public class FeatureGuideView extends FrameLayout {
    private static final float ARROW_HEAD_HALF_WIDTH_DP = 8.4f;
    private static final float ARROW_HEAD_HEIGHT_DP = 15.0f;
    private static final float ARROW_HEIGHT_DP = 40.0f;
    private static final float ARROW_LINE_WIDTH_DP = 4.0f;
    private static final float ARROW_TIP_ANGLE_DEG = 30.0f;
    private static final float ARROW_TIP_GAP_DP = 16.0f;
    private static final float ARROW_VISUAL_SHIFT_DP = 8.0f;
    private static final int GRADIENT_END = -54547;
    private static final int GRADIENT_START = -8967425;
    private static final float PANEL_SIDE_MARGIN_DP = 40.0f;
    private float arrowBaseX;
    private float arrowBaseY;
    private float arrowControl1X;
    private float arrowControl1Y;
    private float arrowControl2X;
    private float arrowControl2Y;
    private float arrowEndX;
    private float arrowEndY;
    private final Paint arrowGlowPaint;
    private final Paint arrowPaint;
    private float arrowStartX;
    private float arrowStartY;
    private float arrowTargetX;
    private float arrowTargetY;
    private boolean arrowVisible;
    private TextView btn_feature_guide_got_it;
    private AppCompatCheckBox chk_feature_guide_no_more;
    private int[] colors;
    private OnDismissedListener dismissedListener;
    private boolean hasManualRect;
    private float holePaddingDp;
    private final Paint holePaint;
    private float holeRadiusDp;
    private final RectF holeRect;
    private final RectF manualRect;
    private final Paint maskPaint;
    private View panelView;
    private float[] positions;
    private final Paint ringPaint;
    private final int[] selfLocation;
    private final View.OnLayoutChangeListener targetLayoutListener;
    private final int[] targetLocation;
    private View targetView;
    private TextView txt_feature_guide_content;
    private TextView txt_feature_guide_desc;
    private GuideButton txt_feature_guide_label;
    private TextView txt_feature_guide_title;

    public interface OnDismissedListener {
        void onDismissed(boolean z);
    }

    public FeatureGuideView(Context context) {
        this(context, null);
    }

    public FeatureGuideView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.colors = new int[]{Color.rgb(105, 158, 255), Color.rgb(FacebookRequestErrorClassification.EC_INVALID_TOKEN, 207, 255), -1, Color.rgb(255, 225, 255), Color.rgb(213, 137, 255)};
        this.positions = new float[]{0.0f, 0.22f, 0.48f, 0.7f, 1.0f};
        Paint paint = new Paint();
        this.maskPaint = paint;
        Paint paint2 = new Paint();
        this.holePaint = paint2;
        Paint paint3 = new Paint();
        this.ringPaint = paint3;
        Paint paint4 = new Paint();
        this.arrowPaint = paint4;
        Paint paint5 = new Paint(1);
        this.arrowGlowPaint = paint5;
        this.holeRect = new RectF();
        this.targetLocation = new int[2];
        this.selfLocation = new int[2];
        this.arrowVisible = false;
        this.holeRadiusDp = 0.0f;
        this.holePaddingDp = 8.0f;
        this.hasManualRect = false;
        this.manualRect = new RectF();
        this.targetLayoutListener = new View.OnLayoutChangeListener() { // from class: com.hismith.hismithapp.view.FeatureGuideView$$ExternalSyntheticLambda0
            @Override // android.view.View.OnLayoutChangeListener
            public final void onLayoutChange(View view, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
                FeatureGuideView.this.m628lambda$new$0$comhismithhismithappviewFeatureGuideView(view, i, i2, i3, i4, i5, i6, i7, i8);
            }
        };
        setClickable(true);
        setFocusable(true);
        setWillNotDraw(false);
        setLayerType(1, null);
        paint.setColor(-939524096);
        paint.setAntiAlias(true);
        paint2.setColor(ViewCompat.MEASURED_STATE_MASK);
        paint2.setAntiAlias(true);
        paint2.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        paint3.setStyle(Paint.Style.STROKE);
        paint3.setStrokeWidth(dp(1.5f));
        paint3.setColor(GRADIENT_END);
        paint3.setAntiAlias(true);
        paint4.setAntiAlias(true);
        paint4.setStrokeWidth(dp(ARROW_LINE_WIDTH_DP));
        paint4.setStrokeCap(Paint.Cap.ROUND);
        paint5.setStyle(Paint.Style.FILL);
        paint5.setColor(Color.rgb(125, 42, 255));
        LayoutInflater.from(context).inflate(R.layout.view_feature_guide, (ViewGroup) this, true);
        this.panelView = findViewById(R.id.layout_feature_guide);
        this.txt_feature_guide_label = (GuideButton) findViewById(R.id.txt_feature_guide_label);
        this.txt_feature_guide_title = (TextView) findViewById(R.id.txt_feature_guide_title);
        this.txt_feature_guide_content = (TextView) findViewById(R.id.txt_feature_guide_content);
        this.txt_feature_guide_desc = (TextView) findViewById(R.id.txt_feature_guide_desc);
        this.btn_feature_guide_got_it = (TextView) findViewById(R.id.btn_feature_guide_got_it);
        this.chk_feature_guide_no_more = (AppCompatCheckBox) findViewById(R.id.chk_feature_guide_no_more);
        LanguageUtil.setText(getContext(), this.txt_feature_guide_label, "featureGuide");
        LanguageUtil.setText(getContext(), this.txt_feature_guide_title, "tapIndicatedIcon");
        LanguageUtil.setText(getContext(), this.txt_feature_guide_content, "safetyAndTracking");
        LanguageUtil.setText(getContext(), this.txt_feature_guide_desc, "safetyRangeDescription");
        LanguageUtil.setText(getContext(), this.btn_feature_guide_got_it, "gotIt");
        LanguageUtil.setText(getContext(), this.chk_feature_guide_no_more, "dontShowAgain");
        this.btn_feature_guide_got_it.setBackground(new GradientGlowDrawable(dp(24.0f), dp(8.0f), dp(0.0f)));
        this.btn_feature_guide_got_it.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.view.FeatureGuideView$$ExternalSyntheticLambda1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                FeatureGuideView.this.m629lambda$new$1$comhismithhismithappviewFeatureGuideView(view);
            }
        });
    }

    private void applyTitleGradient() {
        if (this.txt_feature_guide_title.getWidth() <= 0) {
            return;
        }
        this.txt_feature_guide_title.getPaint().setShader(new LinearGradient(0.0f, 0.0f, this.txt_feature_guide_title.getWidth(), 0.0f, this.colors, this.positions, Shader.TileMode.CLAMP));
        this.txt_feature_guide_title.getPaint().setShadowLayer(12.0f, 0.0f, 0.0f, Color.argb(180, AppInfoTableDecoder.APPLICATION_INFORMATION_TABLE_ID, 76, 255));
        this.txt_feature_guide_title.invalidate();
    }

    private Path buildArrowHead() {
        float f = this.arrowTargetX - this.arrowEndX;
        float f2 = this.arrowTargetY - this.arrowEndY;
        float hypot = (float) Math.hypot(f, f2);
        if (hypot < 0.001f) {
            f = 0.0f;
            f2 = -1.0f;
            hypot = 1.0f;
        }
        float f3 = f / hypot;
        float f4 = -(f2 / hypot);
        float dp = dp(ARROW_HEAD_HALF_WIDTH_DP);
        Path path = new Path();
        path.moveTo(this.arrowEndX, this.arrowEndY);
        float f5 = f4 * dp;
        float f6 = f3 * dp;
        path.lineTo(this.arrowBaseX - f5, this.arrowBaseY - f6);
        path.lineTo(this.arrowBaseX + f5, this.arrowBaseY + f6);
        path.close();
        return path;
    }

    private Path buildArrowRibbon() {
        float[] fArr = new float[48];
        float[] fArr2 = new float[48];
        for (int i = 0; i < 48; i++) {
            float f = i / 47.0f;
            float f2 = 1.0f - f;
            float f3 = f2 * f2 * f2;
            float f4 = 3.0f * f2;
            float f5 = f2 * f4 * f;
            float f6 = f4 * f * f;
            float f7 = f * f * f;
            fArr[i] = (this.arrowStartX * f3) + (this.arrowControl1X * f5) + (this.arrowControl2X * f6) + (this.arrowBaseX * f7);
            fArr2[i] = (f3 * this.arrowStartY) + (f5 * this.arrowControl1Y) + (f6 * this.arrowControl2Y) + (f7 * this.arrowBaseY);
        }
        float dp = dp(ARROW_LINE_WIDTH_DP) / 2.0f;
        Path path = new Path();
        for (int i2 = 0; i2 < 48; i2++) {
            float f8 = i2 / 47.0f;
            float taperLineWidth = taperLineWidth(f8, dp);
            float[] curveNormalAt = curveNormalAt(f8);
            float f9 = fArr[i2] + (curveNormalAt[0] * taperLineWidth);
            float f10 = fArr2[i2] + (curveNormalAt[1] * taperLineWidth);
            if (i2 == 0) {
                path.moveTo(f9, f10);
            } else {
                path.lineTo(f9, f10);
            }
        }
        for (int i3 = 47; i3 >= 0; i3--) {
            float f11 = i3 / 47.0f;
            float taperLineWidth2 = taperLineWidth(f11, dp);
            float[] curveNormalAt2 = curveNormalAt(f11);
            path.lineTo(fArr[i3] - (curveNormalAt2[0] * taperLineWidth2), fArr2[i3] - (curveNormalAt2[1] * taperLineWidth2));
        }
        path.close();
        return path;
    }

    private float[] curveNormalAt(float f) {
        float f2 = 1.0f - f;
        float f3 = f2 * 3.0f * f2;
        float f4 = this.arrowControl1X;
        float f5 = (f4 - this.arrowStartX) * f3;
        float f6 = f2 * 6.0f * f;
        float f7 = this.arrowControl2X;
        float f8 = 3.0f * f * f;
        float f9 = f5 + ((f7 - f4) * f6) + ((this.arrowBaseX - f7) * f8);
        float f10 = this.arrowControl1Y;
        float f11 = f3 * (f10 - this.arrowStartY);
        float f12 = this.arrowControl2Y;
        float f13 = f11 + (f6 * (f12 - f10)) + (f8 * (this.arrowBaseY - f12));
        float hypot = (float) Math.hypot(f9, f13);
        return hypot < 0.001f ? new float[]{0.0f, 1.0f} : new float[]{(-f13) / hypot, f9 / hypot};
    }

    private float dp(float f) {
        return f * getResources().getDisplayMetrics().density;
    }

    private void drawArrow(Canvas canvas) {
        if (this.arrowVisible) {
            this.arrowPaint.setShader(new LinearGradient(this.arrowStartX, this.arrowStartY, this.arrowEndX, this.arrowEndY, new int[]{GRADIENT_START, GRADIENT_END, -1}, new float[]{0.5f, 0.88f, 1.0f}, Shader.TileMode.CLAMP));
            this.arrowPaint.setStyle(Paint.Style.FILL);
            Path buildArrowRibbon = buildArrowRibbon();
            Path buildArrowHead = buildArrowHead();
            this.arrowGlowPaint.setShadowLayer(dp(10.0f), 0.0f, 0.0f, Color.argb(FacebookRequestErrorClassification.EC_INVALID_TOKEN, LocationRequestCompat.QUALITY_LOW_POWER, 35, 255));
            canvas.drawPath(buildArrowRibbon, this.arrowGlowPaint);
            canvas.drawPath(buildArrowHead, this.arrowGlowPaint);
            canvas.drawPath(buildArrowRibbon, this.arrowPaint);
            canvas.drawPath(buildArrowHead, this.arrowPaint);
        }
    }

    private float taperLineWidth(float f, float f2) {
        return f2 * Math.min(1.0f, f / 0.72f);
    }

    private void updateAll() {
        updateHole();
        updatePanelPosition();
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

    private void updatePanelPosition() {
        int max;
        boolean z;
        if (this.panelView == null || this.targetView == null || !isAttachedToWindow() || this.targetView.getVisibility() != 0 || this.targetView.getWidth() <= 0 || this.targetView.getHeight() <= 0) {
            return;
        }
        int width = getWidth();
        int height = getHeight();
        if (width <= 0 || height <= 0) {
            return;
        }
        this.targetView.getLocationInWindow(this.targetLocation);
        getLocationInWindow(this.selfLocation);
        int dp = (int) dp(12.0f);
        int dp2 = width - ((int) dp(80.0f));
        if (dp2 <= 0) {
            return;
        }
        this.panelView.measure(View.MeasureSpec.makeMeasureSpec(dp2, 1073741824), View.MeasureSpec.makeMeasureSpec(0, 0));
        int measuredHeight = this.panelView.getMeasuredHeight();
        if (measuredHeight <= 0) {
            return;
        }
        int[] iArr = this.targetLocation;
        int i = iArr[0];
        int[] iArr2 = this.selfLocation;
        int i2 = i - iArr2[0];
        int i3 = iArr[1] - iArr2[1];
        this.targetView.getWidth();
        int height2 = this.targetView.getHeight() + i3;
        int width2 = i2 + (this.targetView.getWidth() / 2);
        float dp3 = dp(this.holePaddingDp);
        float f = this.holeRect.isEmpty() ? i3 - dp3 : this.holeRect.top;
        float f2 = this.holeRect.isEmpty() ? height2 + dp3 : this.holeRect.bottom;
        float centerX = this.holeRect.isEmpty() ? width2 : this.holeRect.centerX();
        float dp4 = dp(ARROW_TIP_GAP_DP);
        float dp5 = dp(40.0f);
        if (height2 + dp4 + dp5 + measuredHeight <= height) {
            max = (int) (f2 + dp4 + dp5);
            z = true;
        } else {
            max = Math.max(dp, ((int) ((f - dp4) - dp5)) - measuredHeight);
            z = false;
        }
        if (max + measuredHeight > height) {
            max = (height - measuredHeight) - dp;
        }
        float f3 = dp;
        float max2 = Math.max(f3, Math.min((width - dp2) / 2.0f, r2 - dp));
        ViewGroup.MarginLayoutParams marginLayoutParams = (ViewGroup.MarginLayoutParams) this.panelView.getLayoutParams();
        if (marginLayoutParams == null) {
            return;
        }
        marginLayoutParams.width = dp2;
        marginLayoutParams.leftMargin = (int) max2;
        marginLayoutParams.topMargin = max;
        this.panelView.setLayoutParams(marginLayoutParams);
        float centerY = this.holeRect.isEmpty() ? (i3 + height2) / 2.0f : this.holeRect.centerY();
        float f4 = max2 + (dp2 / 2.0f);
        if (!z) {
            max += measuredHeight;
        }
        float f5 = max;
        float dp6 = z ? (f2 + dp4) - dp(8.0f) : (f - dp4) + dp(8.0f);
        float f6 = z ? dp6 - centerY : centerY - dp6;
        float f7 = 0.0f;
        float max3 = f6 > 0.0f ? Math.max(f3, Math.min(centerX - (f6 * ((float) Math.tan(Math.toRadians(30.0d)))), width - dp)) : centerX;
        this.arrowStartX = f4;
        this.arrowStartY = f5;
        this.arrowEndX = max3;
        this.arrowEndY = dp6;
        this.arrowTargetX = centerX;
        this.arrowTargetY = centerY;
        float f8 = centerX - max3;
        float f9 = centerY - dp6;
        float hypot = (float) Math.hypot(f8, f9);
        if (hypot < 0.001f) {
            f9 = -1.0f;
            hypot = 1.0f;
        } else {
            f7 = f8;
        }
        float f10 = f7 / hypot;
        float f11 = f9 / hypot;
        float dp7 = dp(ARROW_HEAD_HEIGHT_DP);
        this.arrowBaseX = max3 - (f10 * dp7);
        this.arrowBaseY = dp6 - (dp7 * f11);
        float min = Math.min(dp(64.0f), ((float) Math.hypot(this.arrowBaseX - this.arrowStartX, this.arrowBaseY - this.arrowStartY)) * 0.28f);
        float f12 = this.arrowStartX;
        float f13 = this.arrowBaseX;
        this.arrowControl1X = f12 + ((f13 - f12) * 0.42f);
        float f14 = this.arrowStartY;
        float f15 = this.arrowBaseY;
        this.arrowControl1Y = f14 + ((f15 - f14) * 0.05f);
        this.arrowControl2X = f13 - (f10 * min);
        this.arrowControl2Y = f15 - (f11 * min);
        this.arrowVisible = true;
        invalidate();
    }

    /* renamed from: lambda$new$0$com-hismith-hismithapp-view-FeatureGuideView, reason: not valid java name */
    /* synthetic */ void m628lambda$new$0$comhismithhismithappviewFeatureGuideView(View view, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
        updateAll();
    }

    /* renamed from: lambda$new$1$com-hismith-hismithapp-view-FeatureGuideView, reason: not valid java name */
    /* synthetic */ void m629lambda$new$1$comhismithhismithappviewFeatureGuideView(View view) {
        OnDismissedListener onDismissedListener = this.dismissedListener;
        if (onDismissedListener != null) {
            onDismissedListener.onDismissed(this.chk_feature_guide_no_more.isChecked());
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
        if (!this.holeRect.isEmpty()) {
            float f = this.holeRadiusDp;
            if (f <= 0.0f) {
                f = 8.0f;
            }
            float dp = dp(f);
            canvas.drawRoundRect(this.holeRect, dp, dp, this.holePaint);
        }
        drawArrow(canvas);
    }

    @Override // android.widget.FrameLayout, android.view.ViewGroup, android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        updateAll();
        applyTitleGradient();
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

    public void setOnDismissedListener(OnDismissedListener onDismissedListener) {
        this.dismissedListener = onDismissedListener;
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
}
