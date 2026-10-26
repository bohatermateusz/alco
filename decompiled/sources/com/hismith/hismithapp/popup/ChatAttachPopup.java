package com.hismith.hismithapp.popup;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.popup.CommonPopupWindow;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class ChatAttachPopup implements View.OnClickListener {
    private Context mContext;
    private View mLineDelete;
    private View mLineReference;
    private View mLineRevoke;
    private int mPosition;
    private TextView mTxtCopy;
    private TextView mTxtDelete;
    private TextView mTxtReference;
    private TextView mTxtRevoke;
    private OnActionClickListener onActionClickListener;
    private CommonPopupWindow popupWindow;
    private View viewGroup;
    private int windowWidth;
    private boolean isShowCopy = true;
    private boolean isShowRevoke = false;
    private boolean isShowDelete = true;

    public interface OnActionClickListener {
        void onCopy(int i);

        void onDelete(int i);

        void onReference(int i);

        void onRevoke(int i);
    }

    public ChatAttachPopup(Context context) {
        this.mContext = context;
        initView();
        initData();
        setListener();
    }

    private static int[] calculatePopWindowPos(View view, View view2) {
        int[] iArr = new int[3];
        int[] iArr2 = new int[2];
        view.getLocationOnScreen(iArr2);
        int height = view.getHeight();
        int screenHeight = MeasureUtil.getScreenHeight(view.getContext());
        MeasureUtil.getScreenWidth(view.getContext());
        view2.measure(0, 0);
        int measuredHeight = view2.getMeasuredHeight();
        view2.getMeasuredWidth();
        int i = iArr2[1];
        if ((screenHeight - i) - height < measuredHeight) {
            iArr[0] = iArr2[0];
            iArr[1] = i - measuredHeight;
            iArr[2] = 1;
            return iArr;
        }
        iArr[0] = iArr2[0];
        iArr[1] = i + height;
        iArr[2] = 0;
        return iArr;
    }

    private void initData() {
    }

    private void initView() {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.popup_chat, (ViewGroup) null, false);
        this.viewGroup = inflate;
        this.mTxtCopy = (TextView) inflate.findViewById(R.id.txt_copy);
        this.mTxtDelete = (TextView) this.viewGroup.findViewById(R.id.txt_delete);
        this.mTxtRevoke = (TextView) this.viewGroup.findViewById(R.id.txt_revoke);
        this.mTxtReference = (TextView) this.viewGroup.findViewById(R.id.txt_reference);
        this.mLineDelete = this.viewGroup.findViewById(R.id.line_delete);
        this.mLineRevoke = this.viewGroup.findViewById(R.id.line_revoke);
        this.mLineReference = this.viewGroup.findViewById(R.id.line_reference);
        LanguageUtil.setText(this.mContext, this.mTxtCopy, "copy");
        LanguageUtil.setText(this.mContext, this.mTxtDelete, "delete");
        LanguageUtil.setText(this.mContext, this.mTxtRevoke, "revoke");
        LanguageUtil.setText(this.mContext, this.mTxtReference, "reference");
    }

    private void setListener() {
        this.mTxtCopy.setOnClickListener(this);
        this.mTxtDelete.setOnClickListener(this);
        this.mTxtRevoke.setOnClickListener(this);
        this.mTxtReference.setOnClickListener(this);
    }

    public void dissPop() {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow != null) {
            commonPopupWindow.dismiss();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        OnActionClickListener onActionClickListener;
        dissPop();
        int id = view.getId();
        if (id == R.id.txt_copy) {
            OnActionClickListener onActionClickListener2 = this.onActionClickListener;
            if (onActionClickListener2 != null) {
                onActionClickListener2.onCopy(this.mPosition);
                return;
            }
            return;
        }
        if (id == R.id.txt_delete) {
            OnActionClickListener onActionClickListener3 = this.onActionClickListener;
            if (onActionClickListener3 != null) {
                onActionClickListener3.onDelete(this.mPosition);
                return;
            }
            return;
        }
        if (id == R.id.txt_revoke) {
            OnActionClickListener onActionClickListener4 = this.onActionClickListener;
            if (onActionClickListener4 != null) {
                onActionClickListener4.onRevoke(this.mPosition);
                return;
            }
            return;
        }
        if (id != R.id.txt_reference || (onActionClickListener = this.onActionClickListener) == null) {
            return;
        }
        onActionClickListener.onReference(this.mPosition);
    }

    public void setOnActionClickListener(OnActionClickListener onActionClickListener) {
        this.onActionClickListener = onActionClickListener;
    }

    public void setShowCopy(boolean z) {
        this.isShowCopy = z;
        TextView textView = this.mTxtCopy;
        if (z) {
            textView.setVisibility(0);
            this.mLineDelete.setVisibility(0);
        } else {
            textView.setVisibility(8);
            this.mLineDelete.setVisibility(8);
        }
    }

    public void setShowDelete(boolean z) {
        this.isShowDelete = z;
        TextView textView = this.mTxtDelete;
        if (z) {
            textView.setVisibility(0);
            this.mLineDelete.setVisibility(0);
        } else {
            textView.setVisibility(8);
            this.mLineDelete.setVisibility(8);
        }
    }

    public void setShowRevoke(boolean z) {
        this.isShowRevoke = z;
        TextView textView = this.mTxtRevoke;
        if (z) {
            textView.setVisibility(0);
            this.mLineRevoke.setVisibility(0);
        } else {
            textView.setVisibility(8);
            this.mLineRevoke.setVisibility(8);
        }
    }

    public void showPop(int i, View view, int i2) {
        this.mPosition = i;
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow == null || !commonPopupWindow.isShowing()) {
            this.popupWindow = new CommonPopupWindow.Builder(this.mContext).setView(this.viewGroup).setWidthAndHeight(-2, -2).setOutsideTouchable(true).create();
            int[] calculatePopWindowPos = calculatePopWindowPos(view, this.viewGroup);
            LogUtil.e("positions[0]:" + calculatePopWindowPos[0] + ", positions[1]:" + calculatePopWindowPos[1] + ", positions[2]:" + calculatePopWindowPos[2]);
            int i3 = calculatePopWindowPos[2];
            CommonPopupWindow commonPopupWindow2 = this.popupWindow;
            if (i3 == 1) {
                commonPopupWindow2.setAnimationStyle(R.style.AnimUp);
                calculatePopWindowPos[1] = calculatePopWindowPos[1] + MeasureUtil.dip2px(this.mContext, 5.0f);
            } else {
                commonPopupWindow2.setAnimationStyle(R.style.popmenu_animation);
                calculatePopWindowPos[1] = calculatePopWindowPos[1] - MeasureUtil.dip2px(this.mContext, 5.0f);
            }
            this.popupWindow.showAtLocation(view, 0, (calculatePopWindowPos[0] - MeasureUtil.dip2px(this.mContext, 13.0f)) + i2, calculatePopWindowPos[1]);
        }
    }
}
