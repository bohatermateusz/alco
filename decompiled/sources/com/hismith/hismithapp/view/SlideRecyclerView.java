package com.hismith.hismithapp.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.widget.Scroller;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

/* loaded from: classes.dex */
public class SlideRecyclerView extends RecyclerView {
    private static final int MINIMUM_VELOCITY = 500;
    private int mFirstX;
    private int mFirstY;
    private ViewGroup mLastView;
    private int mLastX;
    private int mMenuWidth;
    private ViewGroup mMoveView;
    private boolean mMoving;
    private Scroller mScroller;
    private int mTouchSlop;
    private VelocityTracker mVelocity;

    public SlideRecyclerView(Context context) {
        super(context);
        init();
    }

    public SlideRecyclerView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        init();
    }

    public SlideRecyclerView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        init();
    }

    private void addVelocityEvent(MotionEvent motionEvent) {
        VelocityTracker velocityTracker = this.mVelocity;
        if (velocityTracker == null) {
            velocityTracker = VelocityTracker.obtain();
            this.mVelocity = velocityTracker;
        }
        velocityTracker.addMovement(motionEvent);
    }

    private void init() {
        this.mTouchSlop = ViewConfiguration.get(getContext()).getScaledTouchSlop();
        this.mScroller = new Scroller(getContext());
    }

    private boolean isInWindow(View view) {
        if (!(getLayoutManager() instanceof LinearLayoutManager)) {
            return true;
        }
        LinearLayoutManager linearLayoutManager = (LinearLayoutManager) getLayoutManager();
        int findFirstVisibleItemPosition = linearLayoutManager.findFirstVisibleItemPosition();
        int findLastVisibleItemPosition = linearLayoutManager.findLastVisibleItemPosition();
        int position = linearLayoutManager.getPosition(view);
        return position >= findFirstVisibleItemPosition && position <= findLastVisibleItemPosition;
    }

    private void releaseVelocity() {
        VelocityTracker velocityTracker = this.mVelocity;
        if (velocityTracker != null) {
            velocityTracker.clear();
            this.mVelocity.recycle();
            this.mVelocity = null;
        }
    }

    public void closeMenu() {
        ViewGroup viewGroup = this.mLastView;
        if (viewGroup != null) {
            this.mScroller.startScroll(viewGroup.getScrollX(), 0, -this.mLastView.getScrollX(), 0, 500);
            invalidate();
        }
    }

    public void closeMenuNow() {
        ViewGroup viewGroup = this.mLastView;
        if (viewGroup == null || viewGroup.getScrollX() == 0) {
            return;
        }
        this.mLastView.scrollTo(0, 0);
    }

    @Override // android.view.View
    public void computeScroll() {
        if (this.mScroller.computeScrollOffset()) {
            if (isInWindow(this.mLastView)) {
                this.mLastView.scrollTo(this.mScroller.getCurrX(), 0);
                invalidate();
            } else {
                this.mScroller.abortAnimation();
                this.mLastView.scrollTo(this.mScroller.getFinalX(), 0);
            }
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:6:0x001a, code lost:
    
        if (r2 != 3) goto L37;
     */
    @Override // androidx.recyclerview.widget.RecyclerView, android.view.ViewGroup
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        int x = (int) motionEvent.getX();
        int y = (int) motionEvent.getY();
        addVelocityEvent(motionEvent);
        int action = motionEvent.getAction();
        if (action != 0) {
            if (action != 1) {
                if (action == 2) {
                    this.mVelocity.computeCurrentVelocity(1000);
                    int abs = (int) Math.abs(this.mVelocity.getXVelocity());
                    int abs2 = (int) Math.abs(this.mVelocity.getYVelocity());
                    int abs3 = Math.abs(x - this.mFirstX);
                    int abs4 = Math.abs(y - this.mFirstY);
                    if (((Math.abs(abs) >= 500 && abs > abs2) || (abs3 > abs4 && abs3 > this.mTouchSlop)) && this.mMenuWidth > 0 && getScrollState() == 0) {
                        this.mMoving = true;
                        return true;
                    }
                }
            }
            releaseVelocity();
            closeMenuNow();
        } else {
            if (!this.mScroller.isFinished()) {
                this.mScroller.abortAnimation();
            }
            this.mFirstX = x;
            this.mFirstY = y;
            this.mLastX = x;
            ViewGroup viewGroup = (ViewGroup) findChildViewUnder(x, y);
            this.mMoveView = viewGroup;
            ViewGroup viewGroup2 = this.mLastView;
            if (viewGroup2 != null && viewGroup2 != viewGroup && viewGroup2.getScrollX() != 0) {
                closeMenu();
            }
            ViewGroup viewGroup3 = this.mMoveView;
            if (viewGroup3 == null || viewGroup3.getChildCount() != 2) {
                this.mMenuWidth = -1;
            } else {
                this.mMenuWidth = this.mMoveView.getChildAt(1).getWidth();
            }
        }
        return super.onInterceptTouchEvent(motionEvent);
    }

    /* JADX WARN: Code restructure failed: missing block: B:6:0x001b, code lost:
    
        if (r2 != 3) goto L57;
     */
    @Override // androidx.recyclerview.widget.RecyclerView, android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int x = (int) motionEvent.getX();
        int y = (int) motionEvent.getY();
        addVelocityEvent(motionEvent);
        int action = motionEvent.getAction();
        if (action != 1) {
            if (action == 2) {
                if (this.mMoving) {
                    int i = this.mLastX - x;
                    if (this.mMoveView.getScrollX() + i >= 0 && this.mMoveView.getScrollX() + i <= this.mMenuWidth) {
                        this.mMoveView.scrollBy(i, 0);
                    }
                    this.mLastX = x;
                    return true;
                }
                this.mVelocity.computeCurrentVelocity(1000);
                int abs = (int) Math.abs(this.mVelocity.getXVelocity());
                int abs2 = (int) Math.abs(this.mVelocity.getYVelocity());
                int abs3 = Math.abs(x - this.mFirstX);
                int abs4 = Math.abs(y - this.mFirstY);
                if (((Math.abs(abs) >= 500 && abs > abs2) || (abs3 > abs4 && abs3 > this.mTouchSlop)) && this.mMenuWidth > 0 && getScrollState() == 0) {
                    int i2 = this.mLastX - x;
                    if (this.mMoveView.getScrollX() + i2 >= 0 && this.mMoveView.getScrollX() + i2 <= this.mMenuWidth) {
                        this.mMoveView.scrollBy(i2, 0);
                    }
                    this.mLastX = x;
                    this.mMoving = true;
                    return true;
                }
            }
            return super.onTouchEvent(motionEvent);
        }
        if (this.mMoving) {
            if (!this.mScroller.isFinished()) {
                this.mScroller.abortAnimation();
                this.mLastView.scrollTo(this.mScroller.getFinalX(), 0);
            }
            this.mMoving = false;
            this.mLastView = this.mMoveView;
            this.mVelocity.computeCurrentVelocity(1000);
            int scrollX = this.mLastView.getScrollX();
            if (this.mVelocity.getXVelocity() >= 500.0f) {
                this.mScroller.startScroll(scrollX, 0, -scrollX, 0, Math.abs(scrollX));
            } else {
                float xVelocity = this.mVelocity.getXVelocity();
                int i3 = this.mMenuWidth;
                if (xVelocity <= -500.0f) {
                    int i4 = i3 - scrollX;
                    this.mScroller.startScroll(scrollX, 0, i4, 0, Math.abs(i4));
                } else {
                    int i5 = i3 / 2;
                    Scroller scroller = this.mScroller;
                    if (scrollX > i5) {
                        int i6 = i3 - scrollX;
                        scroller.startScroll(scrollX, 0, i6, 0, Math.abs(i6));
                    } else {
                        scroller.startScroll(scrollX, 0, -scrollX, 0, Math.abs(scrollX));
                    }
                }
            }
            invalidate();
        } else {
            ViewGroup viewGroup = this.mLastView;
            if (viewGroup != null && viewGroup.getScrollX() != 0) {
                closeMenu();
            }
        }
        releaseVelocity();
        return super.onTouchEvent(motionEvent);
    }
}
