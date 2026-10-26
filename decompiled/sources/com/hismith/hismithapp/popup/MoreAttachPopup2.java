package com.hismith.hismithapp.popup;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.popup.CommonPopupWindow;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class MoreAttachPopup2 {
    private String mContent;
    private Context mContext;
    private TextView mTxtContent;
    private CommonPopupWindow popupWindow;
    private View viewGroup;
    private int windowWidth;

    public MoreAttachPopup2(Context context, String str) {
        this.mContext = context;
        this.mContent = str;
        initView();
    }

    private void initView() {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.popup_more, (ViewGroup) null, false);
        this.viewGroup = inflate;
        TextView textView = (TextView) inflate.findViewById(R.id.popup_tip_txt_content);
        this.mTxtContent = textView;
        textView.setText(this.mContent);
        this.viewGroup.measure(0, 0);
        this.windowWidth = this.viewGroup.getMeasuredWidth();
    }

    public void dismiss() {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow == null || !commonPopupWindow.isShowing()) {
            return;
        }
        this.popupWindow.dismiss();
        this.popupWindow = null;
    }

    public void showDownPop(View view) {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow == null || !commonPopupWindow.isShowing()) {
            this.popupWindow = new CommonPopupWindow.Builder(this.mContext).setView(this.viewGroup).setWidthAndHeight(-2, -2).setAnimationStyle(R.style.popmenu_animation).setBackGroundLevel(0.5f).setOutsideTouchable(true).create();
            int measuredWidth = view.getMeasuredWidth();
            int[] iArr = new int[2];
            view.getLocationOnScreen(iArr);
            int dip2px = (iArr[0] - this.windowWidth) + (measuredWidth / 2) + MeasureUtil.dip2px(this.mContext, 18.0f);
            LogUtil.e("positions[0]：" + iArr[0] + ", windowWidth:" + this.windowWidth + ", viewWidth:" + measuredWidth + " = " + dip2px);
            this.popupWindow.showAtLocation(view, 0, dip2px, (iArr[1] + view.getHeight()) - MeasureUtil.dip2px(this.mContext, 7.0f));
        }
    }
}
