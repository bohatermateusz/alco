package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public abstract class SelectFollowDialog extends Dialog implements View.OnClickListener {
    private int follow;
    private int isBlocked;
    private Activity mActivity;
    private TextView mTxtBlack;
    private TextView mTxtCancel;
    private TextView mTxtFollow;
    private TextView mTxtNickName;
    private TextView mTxtReport;

    public SelectFollowDialog(Activity activity) {
        super(activity, R.style.ActionSheetDialogStyle);
        this.follow = 0;
        this.isBlocked = 0;
        this.mActivity = activity;
    }

    private void setViewLocation() {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        this.mActivity.getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int i = displayMetrics.heightPixels;
        WindowManager.LayoutParams attributes = getWindow().getAttributes();
        attributes.x = 0;
        attributes.y = i;
        attributes.width = -1;
        attributes.height = -2;
        onWindowAttributesChanged(attributes);
    }

    public abstract void btnClickBlack();

    public abstract void btnClickFollow();

    public abstract void btnClickNickName();

    public abstract void btnClickReport();

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.dialog_follow_txt_nickName) {
            btnClickNickName();
        } else if (id == R.id.dialog_follow_txt_follow) {
            btnClickFollow();
        } else if (id == R.id.dialog_follow_txt_black) {
            btnClickBlack();
        } else if (id == R.id.dialog_follow_txt_report) {
            btnClickReport();
        } else {
            int i = R.id.dialog_follow_txt_cancel;
        }
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_select_follow);
        this.mTxtNickName = (TextView) findViewById(R.id.dialog_follow_txt_nickName);
        this.mTxtFollow = (TextView) findViewById(R.id.dialog_follow_txt_follow);
        this.mTxtBlack = (TextView) findViewById(R.id.dialog_follow_txt_black);
        this.mTxtReport = (TextView) findViewById(R.id.dialog_follow_txt_report);
        this.mTxtCancel = (TextView) findViewById(R.id.dialog_follow_txt_cancel);
        LanguageUtil.setText(this.mActivity, this.mTxtNickName, "remarks");
        LogUtil.e("follow:" + this.follow);
        int i = this.follow;
        Activity activity = this.mActivity;
        if (i == 1) {
            LanguageUtil.setText(activity, this.mTxtFollow, "unsubscribe");
        } else {
            LanguageUtil.setText(activity, this.mTxtFollow, "followCount");
        }
        int i2 = this.isBlocked;
        Activity activity2 = this.mActivity;
        if (i2 == 1) {
            LanguageUtil.setText(activity2, this.mTxtBlack, "removeFromBlocklist");
        } else {
            LanguageUtil.setText(activity2, this.mTxtBlack, "addToBlackList");
        }
        LanguageUtil.setText(this.mActivity, this.mTxtReport, "report");
        LanguageUtil.setText(this.mActivity, this.mTxtCancel, "cancel");
        this.mTxtNickName.setOnClickListener(this);
        this.mTxtFollow.setOnClickListener(this);
        this.mTxtBlack.setOnClickListener(this);
        this.mTxtReport.setOnClickListener(this);
        this.mTxtCancel.setOnClickListener(this);
        setViewLocation();
        setCanceledOnTouchOutside(true);
    }

    public void setFollow(int i) {
        this.follow = i;
    }

    public void setIsBlocked(int i) {
        this.isBlocked = i;
    }
}
