package com.hismith.hismithapp.view;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Handler;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.LinearInterpolator;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class CaptureButton extends View {
    private static final int LONG_PRESS_TIMEOUT = 200;
    private RectF arcRect;
    private Paint bgPaint;
    private int borderColor;
    private Paint borderPaint;
    private float currentProgress;
    private int defaultBgColor;
    private float defaultRadius;
    private float downX;
    private float downY;
    private float innerCircleRadius;
    private Paint innerRedPaint;
    private boolean isLongPressTriggered;
    private boolean isRecording;
    private OnCaptureListener listener;
    private Handler longPressHandler;
    private Runnable longPressRunnable;
    private long maxDuration;
    private ValueAnimator progressAnimator;
    private int progressColor;
    private Paint progressPaint;
    private float progressRadius;
    private int redColor;
    private float strokeWidth;
    private float touchSlop;

    public interface OnCaptureListener {
        void onRecordProgress(long j, float f);

        void onStartRecord();

        void onStopRecord(boolean z);

        void onTakePhoto();
    }

    public CaptureButton(Context context) {
        this(context, null);
    }

    public CaptureButton(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public CaptureButton(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.strokeWidth = MeasureUtil.dip2px(getContext(), 4.0f);
        this.defaultBgColor = Color.parseColor("#E8FFFFFF");
        this.borderColor = Color.parseColor("#80FFFFFF");
        this.redColor = Color.parseColor("#FF3B30");
        this.progressColor = Color.parseColor("#FF3B30");
        this.isRecording = false;
        this.isLongPressTriggered = false;
        this.currentProgress = 0.0f;
        this.longPressHandler = new Handler();
        this.maxDuration = 15000L;
        init();
    }

    private void cancelLongPressTask() {
        this.longPressHandler.removeCallbacks(this.longPressRunnable);
    }

    private void init() {
        this.touchSlop = getResources().getDisplayMetrics().density * 8.0f;
        Paint paint = new Paint(1);
        this.bgPaint = paint;
        paint.setStyle(Paint.Style.FILL);
        this.bgPaint.setColor(this.defaultBgColor);
        Paint paint2 = new Paint(1);
        this.innerRedPaint = paint2;
        paint2.setStyle(Paint.Style.FILL);
        this.innerRedPaint.setColor(this.redColor);
        Paint paint3 = new Paint(1);
        this.borderPaint = paint3;
        paint3.setStyle(Paint.Style.STROKE);
        this.borderPaint.setStrokeWidth(2.0f);
        this.borderPaint.setColor(this.borderColor);
        Paint paint4 = new Paint(1);
        this.progressPaint = paint4;
        paint4.setStyle(Paint.Style.STROKE);
        this.progressPaint.setStrokeWidth(this.strokeWidth);
        this.progressPaint.setColor(this.progressColor);
        this.progressPaint.setStrokeCap(Paint.Cap.ROUND);
        this.arcRect = new RectF();
        this.longPressRunnable = new Runnable() { // from class: com.hismith.hismithapp.view.CaptureButton$$ExternalSyntheticLambda1
            @Override // java.lang.Runnable
            public final void run() {
                CaptureButton.this.m626lambda$init$0$comhismithhismithappviewCaptureButton();
            }
        };
    }

    private void startProgressAnimation() {
        ValueAnimator valueAnimator = this.progressAnimator;
        if (valueAnimator != null) {
            valueAnimator.cancel();
        }
        ValueAnimator ofFloat = ValueAnimator.ofFloat(0.0f, 1.0f);
        this.progressAnimator = ofFloat;
        ofFloat.setDuration(this.maxDuration);
        this.progressAnimator.setInterpolator(new LinearInterpolator());
        this.progressAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.CaptureButton$$ExternalSyntheticLambda0
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public final void onAnimationUpdate(ValueAnimator valueAnimator2) {
                CaptureButton.this.m627xa674f4a2(valueAnimator2);
            }
        });
        this.progressAnimator.addListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.view.CaptureButton.1
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                if (CaptureButton.this.isRecording) {
                    CaptureButton.this.stopRecording(true);
                }
            }
        });
        this.progressAnimator.start();
    }

    private void startRecording() {
        if (this.isRecording) {
            return;
        }
        this.isRecording = true;
        this.currentProgress = 0.0f;
        invalidate();
        OnCaptureListener onCaptureListener = this.listener;
        if (onCaptureListener != null) {
            onCaptureListener.onStartRecord();
            this.listener.onRecordProgress(0L, 0.0f);
        }
        startProgressAnimation();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void stopRecording(boolean z) {
        if (this.isRecording) {
            this.isRecording = false;
            this.isLongPressTriggered = false;
            ValueAnimator valueAnimator = this.progressAnimator;
            if (valueAnimator != null) {
                valueAnimator.cancel();
                this.progressAnimator = null;
            }
            this.currentProgress = 0.0f;
            invalidate();
            OnCaptureListener onCaptureListener = this.listener;
            if (onCaptureListener != null) {
                onCaptureListener.onStopRecord(z);
            }
        }
    }

    /* renamed from: lambda$init$0$com-hismith-hismithapp-view-CaptureButton, reason: not valid java name */
    /* synthetic */ void m626lambda$init$0$comhismithhismithappviewCaptureButton() {
        if (this.isRecording) {
            return;
        }
        this.isLongPressTriggered = true;
        startRecording();
    }

    /* renamed from: lambda$startProgressAnimation$1$com-hismith-hismithapp-view-CaptureButton, reason: not valid java name */
    /* synthetic */ void m627xa674f4a2(ValueAnimator valueAnimator) {
        this.currentProgress = ((Float) valueAnimator.getAnimatedValue()).floatValue();
        invalidate();
        float f = this.maxDuration;
        float f2 = this.currentProgress;
        long j = (long) (f * f2);
        OnCaptureListener onCaptureListener = this.listener;
        if (onCaptureListener != null) {
            onCaptureListener.onRecordProgress(j, f2);
        }
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        float width = getWidth() / 2.0f;
        float height = getHeight() / 2.0f;
        canvas.drawCircle(width, height, this.defaultRadius, this.bgPaint);
        canvas.drawCircle(width, height, this.defaultRadius - 1.0f, this.borderPaint);
        if (this.isRecording || this.isLongPressTriggered) {
            canvas.drawCircle(width, height, this.innerCircleRadius * 0.6f, this.innerRedPaint);
        }
        if (this.isRecording) {
            float f = this.currentProgress;
            if (f > 0.0f) {
                canvas.drawArc(this.arcRect, -90.0f, f * 360.0f, false, this.progressPaint);
            }
        }
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        float min = Math.min(getWidth(), getHeight()) / 2.0f;
        this.defaultRadius = min;
        this.innerCircleRadius = 0.7f * min;
        this.progressRadius = min - (this.strokeWidth / 2.0f);
        this.arcRect.set((getWidth() / 2.0f) - this.progressRadius, (getHeight() / 2.0f) - this.progressRadius, (getWidth() / 2.0f) + this.progressRadius, (getHeight() / 2.0f) + this.progressRadius);
    }

    /* JADX WARN: Code restructure failed: missing block: B:7:0x0018, code lost:
    
        if (r9 != 3) goto L27;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        OnCaptureListener onCaptureListener;
        float x = motionEvent.getX();
        float y = motionEvent.getY();
        int action = motionEvent.getAction();
        if (action == 0) {
            getParent().requestDisallowInterceptTouchEvent(true);
            this.downX = x;
            this.downY = y;
            this.isLongPressTriggered = false;
            this.longPressHandler.postDelayed(this.longPressRunnable, 200L);
            animate().scaleX(0.9f).scaleY(0.9f).setDuration(100L).start();
            performHapticFeedback(268435456);
            return true;
        }
        if (action != 1) {
            if (action == 2) {
                float abs = Math.abs(x - this.downX);
                float abs2 = Math.abs(y - this.downY);
                if (!this.isRecording && !this.isLongPressTriggered) {
                    float f = this.touchSlop;
                    if (abs > f || abs2 > f) {
                        cancelLongPressTask();
                    }
                }
            }
            return true;
        }
        getParent().requestDisallowInterceptTouchEvent(false);
        cancelLongPressTask();
        animate().scaleX(1.0f).scaleY(1.0f).setDuration(100L).start();
        if (this.isRecording) {
            stopRecording(false);
        } else if (!this.isLongPressTriggered && (onCaptureListener = this.listener) != null) {
            onCaptureListener.onTakePhoto();
        }
        return true;
    }

    public void reset() {
        if (this.isRecording) {
            stopRecording(false);
        }
        cancelLongPressTask();
        this.isLongPressTriggered = false;
        this.currentProgress = 0.0f;
        animate().scaleX(1.0f).scaleY(1.0f).cancel();
        setScaleX(1.0f);
        setScaleY(1.0f);
        invalidate();
    }

    public void setMaxDuration(long j) {
        this.maxDuration = j;
    }

    public void setOnCaptureListener(OnCaptureListener onCaptureListener) {
        this.listener = onCaptureListener;
    }
}
