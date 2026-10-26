package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.os.Handler;
import android.os.Message;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class RecordView extends View {
    boolean changeStrokeWidth;
    private float currentRadius;
    private float currentStrokeWidth;
    private boolean down;
    private int downColor;
    private float downRadius;
    private float downX;
    private float downY;
    boolean isAdd;
    private Handler mHandler;
    private OnGestureListener mOnGestureListener;
    private float maxStrokeWidth;
    private float minStrokeWidth;
    private Paint paint;
    private float radiusDis;
    private float slideDis;
    private float strokeWidthDis;
    private int upColor;
    private float upRadius;

    public interface OnGestureListener {
        void onClick();

        void onDown();

        void onUp();
    }

    public RecordView(Context context) {
        super(context);
        this.mHandler = new Handler() { // from class: com.hismith.hismithapp.view.RecordView.1
            @Override // android.os.Handler
            public void handleMessage(Message message) {
                if (RecordView.this.mOnGestureListener != null) {
                    RecordView.this.down = true;
                    RecordView.this.invalidate();
                    RecordView.this.mOnGestureListener.onDown();
                }
            }
        };
        init();
    }

    public RecordView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mHandler = new Handler() { // from class: com.hismith.hismithapp.view.RecordView.1
            @Override // android.os.Handler
            public void handleMessage(Message message) {
                if (RecordView.this.mOnGestureListener != null) {
                    RecordView.this.down = true;
                    RecordView.this.invalidate();
                    RecordView.this.mOnGestureListener.onDown();
                }
            }
        };
        init();
    }

    public RecordView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mHandler = new Handler() { // from class: com.hismith.hismithapp.view.RecordView.1
            @Override // android.os.Handler
            public void handleMessage(Message message) {
                if (RecordView.this.mOnGestureListener != null) {
                    RecordView.this.down = true;
                    RecordView.this.invalidate();
                    RecordView.this.mOnGestureListener.onDown();
                }
            }
        };
        init();
    }

    private void init() {
        this.downColor = R.color.gray9;
        this.upColor = R.color.white;
        Paint paint = new Paint();
        this.paint = paint;
        paint.setAntiAlias(true);
        this.paint.setStyle(Paint.Style.STROKE);
        float dip2px = MeasureUtil.dip2px(getContext(), 10.0f);
        this.currentStrokeWidth = dip2px;
        this.paint.setStrokeWidth(dip2px);
        this.slideDis = MeasureUtil.dip2px(getContext(), 10.0f);
        this.radiusDis = MeasureUtil.dip2px(getContext(), 3.0f);
        this.strokeWidthDis = MeasureUtil.dip2px(getContext(), 1.0f) / 4.0f;
        float f = this.currentStrokeWidth;
        this.minStrokeWidth = f;
        this.maxStrokeWidth = f * 2.0f;
    }

    public void initState() {
        this.down = false;
        this.mHandler.removeMessages(0);
        invalidate();
    }

    public boolean isDown() {
        return this.down;
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        float f;
        super.onDraw(canvas);
        if (!this.down) {
            this.changeStrokeWidth = false;
            float f2 = this.minStrokeWidth;
            this.currentStrokeWidth = f2;
            this.paint.setStrokeWidth(f2);
            this.paint.setColor(ContextCompat.getColor(getContext(), this.upColor));
            float f3 = this.currentRadius;
            float f4 = this.upRadius;
            if (f3 > f4) {
                this.currentRadius = f3 - this.radiusDis;
                invalidate();
            } else if (f3 < f4) {
                this.currentRadius = f4;
                invalidate();
            }
            canvas.drawCircle(getWidth() / 2.0f, getHeight() / 2.0f, this.currentRadius, this.paint);
            return;
        }
        this.paint.setColor(ContextCompat.getColor(getContext(), this.downColor));
        if (this.changeStrokeWidth) {
            boolean z = this.isAdd;
            float f5 = this.currentStrokeWidth;
            if (z) {
                f = f5 + this.strokeWidthDis;
                this.currentStrokeWidth = f;
                if (f > this.maxStrokeWidth) {
                    this.isAdd = false;
                }
            } else {
                f = f5 - this.strokeWidthDis;
                this.currentStrokeWidth = f;
                if (f < this.minStrokeWidth) {
                    this.isAdd = true;
                }
            }
            this.paint.setStrokeWidth(f);
            this.currentRadius = (getWidth() * 0.5f) - this.currentStrokeWidth;
        } else {
            float f6 = this.currentRadius;
            float f7 = this.downRadius;
            if (f6 < f7) {
                this.currentRadius = f6 + this.radiusDis;
            } else if (f6 >= f7) {
                this.currentRadius = f7;
                this.isAdd = true;
                this.changeStrokeWidth = true;
            }
        }
        canvas.drawCircle(getWidth() / 2.0f, getHeight() / 2.0f, this.currentRadius, this.paint);
        invalidate();
    }

    @Override // android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        if (this.downRadius == 0.0f) {
            this.downRadius = (getWidth() * 0.5f) - this.currentStrokeWidth;
            this.upRadius = (getWidth() * 0.3f) - this.currentStrokeWidth;
        }
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        OnGestureListener onGestureListener;
        int action = motionEvent.getAction();
        if (action == 0) {
            ((ViewGroup) getParent()).requestDisallowInterceptTouchEvent(true);
            this.downX = motionEvent.getRawX();
            this.downY = motionEvent.getRawY();
            this.mHandler.sendEmptyMessageDelayed(0, 200L);
        } else if (action == 1) {
            ((ViewGroup) getParent()).requestDisallowInterceptTouchEvent(false);
            float rawX = motionEvent.getRawX();
            float rawY = motionEvent.getRawY();
            if (this.mHandler.hasMessages(0)) {
                this.mHandler.removeMessages(0);
                if (Math.abs(rawX - this.downX) < this.slideDis && Math.abs(rawY - this.downY) < this.slideDis && (onGestureListener = this.mOnGestureListener) != null) {
                    onGestureListener.onClick();
                }
            } else {
                OnGestureListener onGestureListener2 = this.mOnGestureListener;
                if (onGestureListener2 != null) {
                    onGestureListener2.onUp();
                }
            }
            initState();
        }
        return true;
    }

    public void setOnGestureListener(OnGestureListener onGestureListener) {
        this.mOnGestureListener = onGestureListener;
    }
}
