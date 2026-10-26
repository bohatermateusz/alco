package com.hismith.hismithapp.listener;

import android.app.Activity;
import android.graphics.Rect;
import android.view.View;
import android.view.ViewTreeObserver;

/* loaded from: classes3.dex */
public class SoftKeyBoardListener {
    private OnSoftKeyBoardChangeListener onSoftKeyBoardChangeListener;
    private View rootView;
    int rootViewVisibleHeight;

    public interface OnSoftKeyBoardChangeListener {
        void keyBoardHide(int i);

        void keyBoardShow(int i);
    }

    public SoftKeyBoardListener(Activity activity) {
        View decorView = activity.getWindow().getDecorView();
        this.rootView = decorView;
        decorView.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.listener.SoftKeyBoardListener.1
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                Rect rect = new Rect();
                SoftKeyBoardListener.this.rootView.getWindowVisibleDisplayFrame(rect);
                int height = rect.height();
                int i = SoftKeyBoardListener.this.rootViewVisibleHeight;
                SoftKeyBoardListener softKeyBoardListener = SoftKeyBoardListener.this;
                if (i == 0) {
                    softKeyBoardListener.rootViewVisibleHeight = height;
                    return;
                }
                if (softKeyBoardListener.rootViewVisibleHeight == height) {
                    return;
                }
                int i2 = SoftKeyBoardListener.this.rootViewVisibleHeight - height;
                SoftKeyBoardListener softKeyBoardListener2 = SoftKeyBoardListener.this;
                if (i2 > 200) {
                    if (softKeyBoardListener2.onSoftKeyBoardChangeListener != null) {
                        SoftKeyBoardListener.this.onSoftKeyBoardChangeListener.keyBoardShow(SoftKeyBoardListener.this.rootViewVisibleHeight - height);
                    }
                    SoftKeyBoardListener.this.rootViewVisibleHeight = height;
                } else if (height - softKeyBoardListener2.rootViewVisibleHeight > 200) {
                    if (SoftKeyBoardListener.this.onSoftKeyBoardChangeListener != null) {
                        SoftKeyBoardListener.this.onSoftKeyBoardChangeListener.keyBoardHide(height - SoftKeyBoardListener.this.rootViewVisibleHeight);
                    }
                    SoftKeyBoardListener.this.rootViewVisibleHeight = height;
                }
            }
        });
    }

    private void setOnSoftKeyBoardChangeListener(OnSoftKeyBoardChangeListener onSoftKeyBoardChangeListener) {
        this.onSoftKeyBoardChangeListener = onSoftKeyBoardChangeListener;
    }

    public void setListener(OnSoftKeyBoardChangeListener onSoftKeyBoardChangeListener) {
        setOnSoftKeyBoardChangeListener(onSoftKeyBoardChangeListener);
    }
}
