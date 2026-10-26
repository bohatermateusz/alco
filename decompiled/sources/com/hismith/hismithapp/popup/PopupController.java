package com.hismith.hismithapp.popup;

import android.app.Activity;
import android.content.Context;
import android.graphics.drawable.ColorDrawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.PopupWindow;

/* loaded from: classes3.dex */
class PopupController {
    private Context context;
    private int layoutResId;
    View mPopupView;
    private View mView;
    private Window mWindow;
    private PopupWindow popupWindow;

    static class PopupParams {
        public int animationStyle;
        public float bg_level;
        public boolean isShowAnim;
        public boolean isShowBg;
        public boolean isTouchable = true;
        public int layoutResId;
        public Context mContext;
        public int mHeight;
        public View mView;
        public int mWidth;

        public PopupParams(Context context) {
            this.mContext = context;
        }

        public void apply(PopupController popupController) {
            View view = this.mView;
            if (view != null) {
                popupController.setView(view);
            } else {
                int i = this.layoutResId;
                if (i == 0) {
                    throw new IllegalArgumentException("PopupView's contentView is null");
                }
                popupController.setView(i);
            }
            popupController.setWidthAndHeight(this.mWidth, this.mHeight);
            popupController.setOutsideTouchable(this.isTouchable);
            if (this.isShowBg) {
                popupController.setBackGroundLevel(this.bg_level);
            }
            if (this.isShowAnim) {
                popupController.setAnimationStyle(this.animationStyle);
            }
        }
    }

    PopupController(Context context, PopupWindow popupWindow) {
        this.context = context;
        this.popupWindow = popupWindow;
    }

    private void installContent() {
        if (this.layoutResId != 0) {
            this.mPopupView = LayoutInflater.from(this.context).inflate(this.layoutResId, (ViewGroup) null);
        } else {
            View view = this.mView;
            if (view != null) {
                this.mPopupView = view;
            }
        }
        this.popupWindow.setContentView(this.mPopupView);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setAnimationStyle(int i) {
        this.popupWindow.setAnimationStyle(i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setOutsideTouchable(boolean z) {
        this.popupWindow.setBackgroundDrawable(new ColorDrawable(0));
        this.popupWindow.setOutsideTouchable(z);
        this.popupWindow.setFocusable(z);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setWidthAndHeight(int i, int i2) {
        if (i == 0 || i2 == 0) {
            this.popupWindow.setWidth(-2);
            this.popupWindow.setHeight(-2);
        } else {
            this.popupWindow.setWidth(i);
            this.popupWindow.setHeight(i2);
        }
    }

    void setBackGroundLevel(float f) {
        Window window = ((Activity) this.context).getWindow();
        this.mWindow = window;
        WindowManager.LayoutParams attributes = window.getAttributes();
        attributes.alpha = f;
        this.mWindow.setAttributes(attributes);
    }

    public void setView(int i) {
        this.mView = null;
        this.layoutResId = i;
        installContent();
    }

    public void setView(View view) {
        this.mView = view;
        this.layoutResId = 0;
        installContent();
    }
}
