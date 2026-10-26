package com.hismith.hismithapp.view;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.AccelerateDecelerateInterpolator;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class NodeSliderView extends View {
    private static final int MAX_VALUE = 10;
    private static final int MIN_VALUE = 0;
    private static final float NODE_TOUCH_THRESHOLD = 40.0f;
    private static final float THUMB_TOUCH_THRESHOLD = 50.0f;
    private int currentValue;
    private boolean isAnimating;
    private boolean isDragging;
    private int lineColor;
    private float lineEndX;
    private float lineHeight;
    private Paint linePaint;
    private float lineStartX;
    private OnValueChangeListener listener;
    private int nodeColor;
    private Paint nodePaint;
    private float nodeRadius;
    private Paint nodeTextPaint;
    private float[] nodeXPositions;
    private float padding;
    private int progressColor;
    private Paint progressPaint;
    private int selectedNodeColor;
    private Paint selectedNodePaint;
    private float tempThumbX;
    private int tempValue;
    private ValueAnimator thumbAnimator;
    private int thumbColor;
    private Paint thumbPaint;
    private float thumbRadius;
    private Paint thumbTextPaint;
    private float thumbX;

    public interface OnValueChangeListener {
        void onValueChanged(int i);
    }

    public NodeSliderView(Context context) {
        super(context);
        this.lineColor = -7829368;
        this.nodeColor = Color.parseColor("#BDBDBD");
        this.selectedNodeColor = Color.parseColor("#2196F3");
        this.thumbColor = Color.parseColor("#FF4081");
        this.progressColor = Color.parseColor("#2196F3");
        this.lineHeight = 15.0f;
        this.nodeRadius = 12.0f;
        this.thumbRadius = 24.0f;
        this.padding = 24.0f;
        this.currentValue = 3;
        this.nodeXPositions = new float[11];
        this.thumbX = 0.0f;
        this.lineStartX = 0.0f;
        this.lineEndX = 0.0f;
        this.isDragging = false;
        this.isAnimating = false;
        this.tempThumbX = 0.0f;
        this.tempValue = 0;
        init();
    }

    public NodeSliderView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.lineColor = -7829368;
        this.nodeColor = Color.parseColor("#BDBDBD");
        this.selectedNodeColor = Color.parseColor("#2196F3");
        this.thumbColor = Color.parseColor("#FF4081");
        this.progressColor = Color.parseColor("#2196F3");
        this.lineHeight = 15.0f;
        this.nodeRadius = 12.0f;
        this.thumbRadius = 24.0f;
        this.padding = 24.0f;
        this.currentValue = 3;
        this.nodeXPositions = new float[11];
        this.thumbX = 0.0f;
        this.lineStartX = 0.0f;
        this.lineEndX = 0.0f;
        this.isDragging = false;
        this.isAnimating = false;
        this.tempThumbX = 0.0f;
        this.tempValue = 0;
        init();
    }

    public NodeSliderView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.lineColor = -7829368;
        this.nodeColor = Color.parseColor("#BDBDBD");
        this.selectedNodeColor = Color.parseColor("#2196F3");
        this.thumbColor = Color.parseColor("#FF4081");
        this.progressColor = Color.parseColor("#2196F3");
        this.lineHeight = 15.0f;
        this.nodeRadius = 12.0f;
        this.thumbRadius = 24.0f;
        this.padding = 24.0f;
        this.currentValue = 3;
        this.nodeXPositions = new float[11];
        this.thumbX = 0.0f;
        this.lineStartX = 0.0f;
        this.lineEndX = 0.0f;
        this.isDragging = false;
        this.isAnimating = false;
        this.tempThumbX = 0.0f;
        this.tempValue = 0;
        init();
    }

    private void animateToNode(final int i) {
        ValueAnimator valueAnimator;
        if (this.isAnimating || ((valueAnimator = this.thumbAnimator) != null && valueAnimator.isRunning())) {
            this.thumbAnimator.cancel();
        }
        final float f = this.thumbX;
        final float f2 = this.nodeXPositions[i];
        ValueAnimator ofFloat = ValueAnimator.ofFloat(0.0f, 1.0f);
        this.thumbAnimator = ofFloat;
        ofFloat.setDuration(300L);
        this.thumbAnimator.setInterpolator(new AccelerateDecelerateInterpolator());
        this.thumbAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.NodeSliderView.1
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                NodeSliderView.this.isAnimating = true;
                float animatedFraction = valueAnimator2.getAnimatedFraction();
                NodeSliderView nodeSliderView = NodeSliderView.this;
                float f3 = f;
                nodeSliderView.thumbX = f3 + ((f2 - f3) * animatedFraction);
                NodeSliderView nodeSliderView2 = NodeSliderView.this;
                nodeSliderView2.tempValue = nodeSliderView2.calculateNearestNodeValue(nodeSliderView2.thumbX);
                NodeSliderView.this.invalidate();
            }
        });
        this.thumbAnimator.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.view.NodeSliderView.2
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
                NodeSliderView.this.isAnimating = false;
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                NodeSliderView.this.isAnimating = false;
                NodeSliderView.this.setValue(i, true);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                NodeSliderView.this.isAnimating = true;
            }
        });
        this.thumbAnimator.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public int calculateNearestNodeValue(float f) {
        float f2 = Float.MAX_VALUE;
        int i = 0;
        for (int i2 = 0; i2 <= 10; i2++) {
            float abs = Math.abs(f - this.nodeXPositions[i2]);
            if (abs < f2) {
                i = i2;
                f2 = abs;
            }
        }
        return i;
    }

    private void calculateNodePositions() {
        float f = (this.lineEndX - this.lineStartX) / 10.0f;
        for (int i = 0; i <= 10; i++) {
            this.nodeXPositions[i] = this.lineStartX + (i * f);
        }
    }

    private void drawAllNodes(Canvas canvas, float f) {
        float currentThumbX = getCurrentThumbX();
        for (int i = 0; i <= 10; i++) {
            float f2 = this.nodeXPositions[i];
            canvas.drawCircle(f2, f, this.nodeRadius, f2 <= currentThumbX ? this.selectedNodePaint : this.nodePaint);
        }
    }

    private void drawBackgroundLine(Canvas canvas, float f) {
        float f2 = this.lineStartX;
        float f3 = this.lineHeight;
        RectF rectF = new RectF(f2, f - (f3 / 2.0f), this.lineEndX, f + (f3 / 2.0f));
        float f4 = this.lineHeight;
        canvas.drawRoundRect(rectF, f4 / 2.0f, f4 / 2.0f, this.linePaint);
    }

    private void drawNodeText(Canvas canvas, float f, float f2, String str, boolean z) {
        Paint.FontMetrics fontMetrics = this.nodeTextPaint.getFontMetrics();
        float f3 = f2 - ((fontMetrics.ascent + fontMetrics.descent) / 2.0f);
        int color = this.nodeTextPaint.getColor();
        this.nodeTextPaint.setColor(z ? -1 : ViewCompat.MEASURED_STATE_MASK);
        canvas.drawText(str, f, f3, this.nodeTextPaint);
        this.nodeTextPaint.setColor(color);
    }

    private void drawProgressLine(Canvas canvas, float f) {
        canvas.drawLine(this.lineStartX, f, getCurrentThumbX(), f, this.progressPaint);
    }

    private void drawThumb(Canvas canvas, float f) {
        float currentThumbX = getCurrentThumbX();
        canvas.drawCircle(currentThumbX, f, this.thumbRadius, this.thumbPaint);
        Paint.FontMetrics fontMetrics = this.thumbTextPaint.getFontMetrics();
        canvas.drawText(String.valueOf(getCurrentDisplayValue()), currentThumbX, f - ((fontMetrics.ascent + fontMetrics.descent) / 2.0f), this.thumbTextPaint);
    }

    private int getCurrentDisplayValue() {
        return this.isDragging ? this.tempValue : this.isAnimating ? calculateNearestNodeValue(this.thumbX) : this.currentValue;
    }

    private float getCurrentThumbX() {
        return this.isDragging ? this.tempThumbX : this.thumbX;
    }

    private int getTouchedNode(float f, float f2) {
        float abs = Math.abs(f2 - (getHeight() / 2.0f));
        float f3 = NODE_TOUCH_THRESHOLD;
        int i = -1;
        if (abs > NODE_TOUCH_THRESHOLD) {
            return -1;
        }
        for (int i2 = 0; i2 <= 10; i2++) {
            float abs2 = Math.abs(f - this.nodeXPositions[i2]);
            if (abs2 < f3) {
                i = i2;
                f3 = abs2;
            }
        }
        return i;
    }

    private void init() {
        this.lineColor = getContext().getColor(R.color.white10);
        this.nodeColor = getContext().getColor(R.color.play_mode_smooth_line_circle_mark);
        this.selectedNodeColor = getContext().getColor(R.color.play_mode_smooth_line_circle_mark);
        this.thumbColor = getContext().getColor(R.color.play_mode_smooth_line_circle_mark);
        this.progressColor = getContext().getColor(R.color.play_mode_smooth_node_color);
        Paint paint = new Paint(1);
        this.linePaint = paint;
        paint.setColor(this.lineColor);
        this.linePaint.setStyle(Paint.Style.FILL);
        Paint paint2 = new Paint(1);
        this.nodePaint = paint2;
        paint2.setColor(this.nodeColor);
        this.nodePaint.setStyle(Paint.Style.FILL);
        Paint paint3 = new Paint(1);
        this.selectedNodePaint = paint3;
        paint3.setColor(this.selectedNodeColor);
        this.selectedNodePaint.setStyle(Paint.Style.FILL);
        Paint paint4 = new Paint(1);
        this.thumbPaint = paint4;
        paint4.setColor(this.thumbColor);
        this.thumbPaint.setStyle(Paint.Style.FILL);
        this.thumbPaint.setShadowLayer(8.0f, 3.0f, 3.0f, Color.parseColor("#55000000"));
        Paint paint5 = new Paint(1);
        this.progressPaint = paint5;
        paint5.setColor(this.progressColor);
        this.progressPaint.setStyle(Paint.Style.FILL);
        this.progressPaint.setStrokeWidth(this.lineHeight);
        this.progressPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint6 = new Paint(1);
        this.nodeTextPaint = paint6;
        paint6.setColor(-1);
        this.nodeTextPaint.setTextSize(this.nodeRadius * 1.2f);
        this.nodeTextPaint.setTextAlign(Paint.Align.CENTER);
        Paint paint7 = new Paint(1);
        this.thumbTextPaint = paint7;
        paint7.setColor(ViewCompat.MEASURED_STATE_MASK);
        this.thumbTextPaint.setTextSize(this.thumbRadius * 1.5f);
        this.thumbTextPaint.setTextAlign(Paint.Align.CENTER);
        this.thumbTextPaint.setFakeBoldText(true);
        setLayerType(1, null);
    }

    private void moveThumb(float f) {
        float f2 = this.lineStartX;
        if (f < f2) {
            f = f2;
        }
        float f3 = this.lineEndX;
        if (f > f3) {
            f = f3;
        }
        this.tempThumbX = f;
    }

    private void snapToNearestNode() {
        setValue(calculateNearestNodeValue(this.tempThumbX), false);
    }

    private void updateThumbPosition() {
        this.thumbX = this.nodeXPositions[this.currentValue];
    }

    public int getValue() {
        return this.currentValue;
    }

    @Override // android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        ValueAnimator valueAnimator = this.thumbAnimator;
        if (valueAnimator != null) {
            valueAnimator.cancel();
            this.thumbAnimator = null;
        }
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float height = getHeight() / 2.0f;
        drawBackgroundLine(canvas, height);
        drawProgressLine(canvas, height);
        drawAllNodes(canvas, height);
        drawThumb(canvas, height);
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        float f = this.padding;
        this.lineStartX = f;
        this.lineEndX = i - f;
        calculateNodePositions();
        updateThumbPosition();
        this.tempThumbX = this.thumbX;
    }

    /* JADX WARN: Code restructure failed: missing block: B:11:0x001a, code lost:
    
        if (r3 != 3) goto L32;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        if (this.isAnimating) {
            return true;
        }
        float x = motionEvent.getX();
        float y = motionEvent.getY();
        int action = motionEvent.getAction();
        if (action != 0) {
            if (action != 1) {
                if (action == 2) {
                    if (this.isDragging) {
                        moveThumb(x);
                        this.tempValue = calculateNearestNodeValue(this.tempThumbX);
                        invalidate();
                        return true;
                    }
                }
            }
            if (this.isDragging) {
                snapToNearestNode();
                this.isDragging = false;
                OnValueChangeListener onValueChangeListener = this.listener;
                if (onValueChangeListener != null) {
                    onValueChangeListener.onValueChanged(this.currentValue);
                }
                invalidate();
                return true;
            }
        } else {
            int touchedNode = getTouchedNode(x, y);
            if (Math.abs(x - this.thumbX) <= 50.0f) {
                this.isDragging = true;
                this.tempThumbX = this.thumbX;
                this.tempValue = this.currentValue;
                return true;
            }
            if (touchedNode != -1) {
                animateToNode(touchedNode);
                return true;
            }
        }
        return super.onTouchEvent(motionEvent);
    }

    public void setAnimationDuration(long j) {
    }

    public void setNodeColor(int i) {
        this.nodeColor = i;
        this.nodePaint.setColor(i);
        invalidate();
    }

    public void setNodeRadius(float f) {
        this.nodeRadius = f;
        this.nodeTextPaint.setTextSize(f * 1.2f);
        invalidate();
    }

    public void setOnValueChangeListener(OnValueChangeListener onValueChangeListener) {
        this.listener = onValueChangeListener;
    }

    public void setProgressColor(int i) {
        this.progressColor = i;
        this.progressPaint.setColor(i);
        invalidate();
    }

    public void setSelectedNodeColor(int i) {
        this.selectedNodeColor = i;
        this.selectedNodePaint.setColor(i);
        invalidate();
    }

    public void setThumbColor(int i) {
        this.thumbColor = i;
        this.thumbPaint.setColor(i);
        invalidate();
    }

    public void setThumbRadius(float f) {
        this.thumbRadius = f;
        this.padding = f;
        this.thumbTextPaint.setTextSize(f * 0.8f);
        invalidate();
    }

    public void setValue(int i) {
        setValue(i, false);
    }

    public void setValue(int i, boolean z) {
        OnValueChangeListener onValueChangeListener;
        if (i < 0) {
            i = 0;
        }
        if (i > 10) {
            i = 10;
        }
        this.currentValue = i;
        updateThumbPosition();
        this.tempThumbX = this.thumbX;
        this.tempValue = this.currentValue;
        invalidate();
        if (!z || (onValueChangeListener = this.listener) == null) {
            return;
        }
        onValueChangeListener.onValueChanged(this.currentValue);
    }
}
