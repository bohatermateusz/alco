package com.hismith.hismithapp.popup;

import android.content.Context;
import android.view.View;
import android.widget.PopupWindow;
import com.hismith.hismithapp.popup.PopupController;
import com.hismith.hismithapp.utils.CommonUtil;

/* loaded from: classes3.dex */
public class CommonPopupWindow extends PopupWindow {
    final PopupController controller;

    public static class Builder {
        private ViewInterface listener;
        private final PopupController.PopupParams params;

        public Builder(Context context) {
            this.params = new PopupController.PopupParams(context);
        }

        public CommonPopupWindow create() {
            CommonPopupWindow commonPopupWindow = new CommonPopupWindow(this.params.mContext);
            this.params.apply(commonPopupWindow.controller);
            if (this.listener != null && this.params.layoutResId != 0) {
                this.listener.getChildView(commonPopupWindow.controller.mPopupView, this.params.layoutResId);
            }
            CommonUtil.measureWidthAndHeight(commonPopupWindow.controller.mPopupView);
            return commonPopupWindow;
        }

        public Builder setAnimationStyle(int i) {
            this.params.isShowAnim = true;
            this.params.animationStyle = i;
            return this;
        }

        public Builder setBackGroundLevel(float f) {
            this.params.isShowBg = true;
            this.params.bg_level = f;
            return this;
        }

        public Builder setOutsideTouchable(boolean z) {
            this.params.isTouchable = z;
            return this;
        }

        public Builder setView(int i) {
            this.params.mView = null;
            this.params.layoutResId = i;
            return this;
        }

        public Builder setView(View view) {
            this.params.mView = view;
            this.params.layoutResId = 0;
            return this;
        }

        public Builder setViewOnclickListener(ViewInterface viewInterface) {
            this.listener = viewInterface;
            return this;
        }

        public Builder setWidthAndHeight(int i, int i2) {
            this.params.mWidth = i;
            this.params.mHeight = i2;
            return this;
        }
    }

    public interface ViewInterface {
        void getChildView(View view, int i);
    }

    private CommonPopupWindow(Context context) {
        this.controller = new PopupController(context, this);
    }

    @Override // android.widget.PopupWindow
    public void dismiss() {
        super.dismiss();
        this.controller.setBackGroundLevel(1.0f);
    }

    @Override // android.widget.PopupWindow
    public int getHeight() {
        return this.controller.mPopupView.getMeasuredHeight();
    }

    @Override // android.widget.PopupWindow
    public int getWidth() {
        return this.controller.mPopupView.getMeasuredWidth();
    }
}
