package com.hismith.hismithapp.utils;

import android.R;
import android.app.Activity;
import android.graphics.Rect;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;

/* loaded from: classes3.dex */
public class KeyBordStateUtil {
    private onKeyBordStateListener listener;
    private int mFirstVisibleHeight;
    private ViewTreeObserver.OnGlobalLayoutListener mOnGlobalLayoutListener = new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.utils.KeyBordStateUtil.1
        @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
        public void onGlobalLayout() {
            KeyBordStateUtil.this.calKeyBordState();
        }
    };
    private int mVisibleHeight;
    private View rootLayout;

    public interface onKeyBordStateListener {
        void onSoftKeyBoardHide();

        void onSoftKeyBoardShow(int i);
    }

    public KeyBordStateUtil(Activity activity) {
        View childAt = ((ViewGroup) activity.findViewById(R.id.content)).getChildAt(0);
        this.rootLayout = childAt;
        childAt.getViewTreeObserver().addOnGlobalLayoutListener(this.mOnGlobalLayoutListener);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void calKeyBordState() {
        Rect rect = new Rect();
        this.rootLayout.getWindowVisibleDisplayFrame(rect);
        int height = rect.height();
        int i = this.mVisibleHeight;
        if (i == 0) {
            this.mVisibleHeight = height;
            this.mFirstVisibleHeight = height;
            return;
        }
        if (i == height) {
            return;
        }
        this.mVisibleHeight = height;
        int i2 = this.mFirstVisibleHeight;
        if (height >= i2) {
            onKeyBordStateListener onkeybordstatelistener = this.listener;
            if (onkeybordstatelistener != null) {
                onkeybordstatelistener.onSoftKeyBoardHide();
                return;
            }
            return;
        }
        int abs = Math.abs(height - i2);
        onKeyBordStateListener onkeybordstatelistener2 = this.listener;
        if (onkeybordstatelistener2 != null) {
            onkeybordstatelistener2.onSoftKeyBoardShow(abs);
        }
    }

    public void addOnKeyBordStateListener(onKeyBordStateListener onkeybordstatelistener) {
        this.listener = onkeybordstatelistener;
    }

    public void removeOnKeyBordStateListener() {
        if (this.rootLayout != null && this.mOnGlobalLayoutListener != null) {
            this.rootLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this.mOnGlobalLayoutListener);
        }
        if (this.listener != null) {
            this.listener = null;
        }
    }
}
