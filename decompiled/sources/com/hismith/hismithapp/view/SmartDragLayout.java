package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewConfiguration;
import android.widget.FrameLayout;
import android.widget.OverScroller;
import androidx.core.view.NestedScrollingParent;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.utils.LogUtil;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.animator.ShadowBgAnimator;
import com.lxj.xpopup.enums.LayoutStatus;
import com.lxj.xpopup.util.XPopupUtils;

/* loaded from: classes3.dex */
public class SmartDragLayout extends FrameLayout implements NestedScrollingParent {
    private static final String TAG = "SmartDragLayout";
    ShadowBgAnimator bgAnimator;
    private View child;
    boolean dismissOnTouchOutside;
    boolean enableDrag;
    boolean hasShadowBg;
    boolean isScrollUp;
    boolean isUserClose;
    int lastHeight;
    private OnCloseListener listener;
    int maxY;
    int minY;
    OverScroller scroller;
    LayoutStatus status;
    float touchX;
    float touchY;
    VelocityTracker tracker;

    public interface OnCloseListener {
        void onClose();

        void onOpen();
    }

    public SmartDragLayout(Context context) {
        this(context, null);
    }

    public SmartDragLayout(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public SmartDragLayout(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.bgAnimator = new ShadowBgAnimator();
        this.enableDrag = true;
        this.dismissOnTouchOutside = true;
        this.hasShadowBg = true;
        this.isUserClose = false;
        this.status = LayoutStatus.Close;
        if (this.enableDrag) {
            this.scroller = new OverScroller(context);
        }
    }

    private void finishScroll() {
        if (this.enableDrag) {
            boolean z = this.isScrollUp;
            int i = this.maxY;
            this.scroller.startScroll(getScrollX(), getScrollY(), 0, (getScrollY() > (z ? i - this.minY : (i - this.minY) * 2) / 3 ? this.maxY : this.minY) - getScrollY(), XPopup.getAnimationDuration());
            ViewCompat.postInvalidateOnAnimation(this);
        }
    }

    public void close() {
        this.isUserClose = true;
        this.status = LayoutStatus.Closing;
        post(new Runnable() { // from class: com.hismith.hismithapp.view.SmartDragLayout.2
            @Override // java.lang.Runnable
            public void run() {
                SmartDragLayout smartDragLayout = SmartDragLayout.this;
                smartDragLayout.smoothScroll(smartDragLayout.minY - SmartDragLayout.this.getScrollY());
            }
        });
    }

    @Override // android.view.View
    public void computeScroll() {
        super.computeScroll();
        if (this.scroller.computeScrollOffset()) {
            scrollTo(this.scroller.getCurrX(), this.scroller.getCurrY());
            ViewCompat.postInvalidateOnAnimation(this);
        }
    }

    public void dismissOnTouchOutside(boolean z) {
        this.dismissOnTouchOutside = z;
    }

    @Override // android.view.ViewGroup, android.view.View
    public boolean dispatchTouchEvent(MotionEvent motionEvent) {
        this.isUserClose = true;
        return super.dispatchTouchEvent(motionEvent);
    }

    public void enableDrag(boolean z) {
        this.enableDrag = z;
    }

    @Override // android.view.ViewGroup, androidx.core.view.NestedScrollingParent
    public int getNestedScrollAxes() {
        return 2;
    }

    public void hasShadowBg(boolean z) {
        this.hasShadowBg = z;
    }

    @Override // android.view.ViewGroup, android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        this.isScrollUp = false;
        this.isUserClose = false;
        setTranslationY(0.0f);
    }

    @Override // android.widget.FrameLayout, android.view.ViewGroup, android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        this.maxY = this.child.getMeasuredHeight();
        this.minY = 0;
        int measuredWidth = (getMeasuredWidth() / 2) - (this.child.getMeasuredWidth() / 2);
        boolean z2 = this.enableDrag;
        View view = this.child;
        if (z2) {
            view.layout(measuredWidth, getMeasuredHeight(), this.child.getMeasuredWidth() + measuredWidth, getMeasuredHeight() + this.maxY);
            if (this.status == LayoutStatus.Open) {
                scrollTo(getScrollX(), getScrollY() - (this.lastHeight - this.maxY));
            }
        } else {
            view.layout(measuredWidth, getMeasuredHeight() - this.child.getMeasuredHeight(), this.child.getMeasuredWidth() + measuredWidth, getMeasuredHeight());
        }
        this.lastHeight = this.maxY;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent, androidx.core.view.NestedScrollingParent
    public boolean onNestedFling(View view, float f, float f2, boolean z) {
        if (getScrollY() <= this.minY || getScrollY() >= this.maxY || f2 >= -1500.0f) {
            return false;
        }
        close();
        return false;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent, androidx.core.view.NestedScrollingParent
    public boolean onNestedPreFling(View view, float f, float f2) {
        return false;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent, androidx.core.view.NestedScrollingParent
    public void onNestedPreScroll(View view, int i, int i2, int[] iArr) {
        if (i2 > 0) {
            int scrollY = getScrollY() + i2;
            if (scrollY < this.maxY) {
                iArr[1] = i2;
            }
            scrollTo(getScrollX(), scrollY);
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent, androidx.core.view.NestedScrollingParent
    public void onNestedScroll(View view, int i, int i2, int i3, int i4) {
        scrollTo(getScrollX(), getScrollY() + i4);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent, androidx.core.view.NestedScrollingParent
    public void onNestedScrollAccepted(View view, View view2, int i) {
        this.scroller.abortAnimation();
    }

    @Override // android.view.ViewGroup, android.view.ViewParent, androidx.core.view.NestedScrollingParent
    public boolean onStartNestedScroll(View view, View view2, int i) {
        return i == 2 && this.enableDrag;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent, androidx.core.view.NestedScrollingParent
    public void onStopNestedScroll(View view) {
        finishScroll();
    }

    /* JADX WARN: Code restructure failed: missing block: B:12:0x001c, code lost:
    
        if (r0 != 3) goto L35;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        if (this.scroller.computeScrollOffset()) {
            this.touchX = 0.0f;
            this.touchY = 0.0f;
            return false;
        }
        int action = motionEvent.getAction();
        if (action != 0) {
            if (action != 1) {
                if (action == 2) {
                    if (this.enableDrag) {
                        this.tracker.addMovement(motionEvent);
                        this.tracker.computeCurrentVelocity(1000);
                        scrollTo(getScrollX(), getScrollY() - ((int) (motionEvent.getY() - this.touchY)));
                        this.touchY = motionEvent.getY();
                    }
                }
            }
            Rect rect = new Rect();
            this.child.getGlobalVisibleRect(rect);
            if (!XPopupUtils.isInRect(motionEvent.getRawX(), motionEvent.getRawY(), rect) && this.dismissOnTouchOutside && ((float) Math.sqrt(Math.pow(motionEvent.getX() - this.touchX, 2.0d) + Math.pow(motionEvent.getY() - this.touchY, 2.0d))) < ViewConfiguration.get(getContext()).getScaledTouchSlop()) {
                performClick();
            }
            if (this.enableDrag) {
                if (this.tracker.getYVelocity() > 1500.0f) {
                    close();
                } else {
                    finishScroll();
                }
                this.tracker.clear();
                this.tracker.recycle();
            }
        } else {
            if (this.enableDrag) {
                this.tracker = VelocityTracker.obtain();
            }
            this.touchX = motionEvent.getX();
            this.touchY = motionEvent.getY();
        }
        return true;
    }

    @Override // android.view.ViewGroup
    public void onViewAdded(View view) {
        super.onViewAdded(view);
        this.child = view;
    }

    public void open() {
        setVisibility(0);
        this.status = LayoutStatus.Opening;
        post(new Runnable() { // from class: com.hismith.hismithapp.view.SmartDragLayout.1
            @Override // java.lang.Runnable
            public void run() {
                SmartDragLayout smartDragLayout = SmartDragLayout.this;
                smartDragLayout.smoothScroll(smartDragLayout.maxY - SmartDragLayout.this.getScrollY());
            }
        });
    }

    @Override // android.view.View
    public void scrollTo(int i, int i2) {
        LogUtil.e("scrollTo x:" + i + ", y:" + i2);
        int i3 = this.maxY;
        if (i2 > i3) {
            i2 = i3;
        }
        int i4 = this.minY;
        if (i2 < i4) {
            i2 = i4;
        }
        float f = ((i2 - i4) * 1.0f) / (i3 - i4);
        this.isScrollUp = i2 > getScrollY();
        if (this.hasShadowBg) {
            setBackgroundColor(this.bgAnimator.calculateBgColor(f));
        }
        if (this.listener != null) {
            if (this.isUserClose && f == 0.0f && this.status != LayoutStatus.Close) {
                this.status = LayoutStatus.Close;
                this.listener.onClose();
                setVisibility(8);
            } else if (f == 1.0f && this.status != LayoutStatus.Open) {
                this.status = LayoutStatus.Open;
                this.listener.onOpen();
            }
        }
        super.scrollTo(i, i2);
    }

    public void setOnCloseListener(OnCloseListener onCloseListener) {
        this.listener = onCloseListener;
    }

    public void smoothScroll(final int i) {
        post(new Runnable() { // from class: com.hismith.hismithapp.view.SmartDragLayout.3
            @Override // java.lang.Runnable
            public void run() {
                LogUtil.e("smoothScroll dy:" + i);
                SmartDragLayout.this.scroller.startScroll(SmartDragLayout.this.getScrollX(), SmartDragLayout.this.getScrollY(), 0, i, XPopup.getAnimationDuration());
                ViewCompat.postInvalidateOnAnimation(SmartDragLayout.this);
            }
        });
    }
}
