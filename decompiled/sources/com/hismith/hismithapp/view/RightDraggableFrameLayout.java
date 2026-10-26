package com.hismith.hismithapp.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewParent;
import android.widget.FrameLayout;

/* loaded from: classes3.dex */
public class RightDraggableFrameLayout extends FrameLayout {
    private int bottomBoundViewId;
    private float downRawY;
    private float initialY;
    private boolean isDragging;
    private int topBoundViewId;
    private int touchSlop;

    public RightDraggableFrameLayout(Context context) {
        super(context);
        this.isDragging = false;
        this.topBoundViewId = -1;
        this.bottomBoundViewId = -1;
        init(context);
    }

    public RightDraggableFrameLayout(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.isDragging = false;
        this.topBoundViewId = -1;
        this.bottomBoundViewId = -1;
        init(context);
    }

    public RightDraggableFrameLayout(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isDragging = false;
        this.topBoundViewId = -1;
        this.bottomBoundViewId = -1;
        init(context);
    }

    private float applyBounds(float f) {
        int height;
        View findViewById;
        View findViewById2;
        View view = (View) getParent();
        if (view != null && (height = view.getHeight()) != 0) {
            int paddingTop = view.getPaddingTop();
            int paddingBottom = view.getPaddingBottom();
            int height2 = getHeight();
            float f2 = paddingTop;
            float f3 = (height - paddingBottom) - height2;
            int i = this.topBoundViewId;
            if (i != -1 && (findViewById2 = view.findViewById(i)) != null && findViewById2.getVisibility() == 0) {
                float bottom = findViewById2.getBottom();
                if (bottom > f2) {
                    f2 = bottom;
                }
            }
            int i2 = this.bottomBoundViewId;
            if (i2 != -1 && (findViewById = view.findViewById(i2)) != null && findViewById.getVisibility() == 0) {
                float top = findViewById.getTop() - height2;
                if (top < f3) {
                    f3 = top;
                }
            }
            if (f2 > f3) {
                return (f2 + f3) / 2.0f;
            }
            if (f < f2) {
                return f2;
            }
            if (f > f3) {
                return f3;
            }
        }
        return f;
    }

    private void init(Context context) {
        this.touchSlop = ViewConfiguration.get(context).getScaledTouchSlop();
        setClickable(true);
    }

    @Override // android.view.ViewGroup
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        return false;
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0) {
            this.downRawY = motionEvent.getRawY();
            this.initialY = getY();
            this.isDragging = false;
            return true;
        }
        if (actionMasked != 1) {
            if (actionMasked == 2) {
                float rawY = motionEvent.getRawY() - this.downRawY;
                float applyBounds = applyBounds(this.initialY + rawY);
                if (!this.isDragging && Math.abs(rawY) > this.touchSlop) {
                    this.isDragging = true;
                    ViewParent parent = getParent();
                    if (parent != null) {
                        parent.requestDisallowInterceptTouchEvent(true);
                    }
                }
                setY(applyBounds);
                return true;
            }
            if (actionMasked != 3) {
                return super.onTouchEvent(motionEvent);
            }
        }
        if (!this.isDragging) {
            performClick();
        }
        this.isDragging = false;
        return true;
    }

    public void setBottomBoundViewId(int i) {
        this.bottomBoundViewId = i;
    }

    public void setTopBoundViewId(int i) {
        this.topBoundViewId = i;
    }
}
