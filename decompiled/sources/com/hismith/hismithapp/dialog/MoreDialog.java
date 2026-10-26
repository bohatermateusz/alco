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

/* loaded from: classes3.dex */
public class MoreDialog extends Dialog implements View.OnClickListener {
    private Activity mActivity;
    private View mDeleteLine;
    private TextView mTxtActionOne;
    private TextView mTxtActionThree;
    private TextView mTxtActionTwo;
    private TextView mTxtCancel;
    private String mTxtOne;
    private String mTxtThree;
    private String mTxtTwo;
    private OnClickItemActionListener onClickItemActionListener;

    public interface OnClickItemActionListener {
        void btnClickOne();

        void btnClickThree();

        void btnClickTwo();
    }

    public MoreDialog(Activity activity, String str, String str2, String str3) {
        super(activity, R.style.ActionSheetDialogStyle);
        this.mActivity = activity;
        this.mTxtOne = str;
        this.mTxtTwo = str2;
        this.mTxtThree = str3;
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

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.dialog_txt_action_one) {
            this.onClickItemActionListener.btnClickOne();
        } else if (id == R.id.dialog_txt_action_two) {
            this.onClickItemActionListener.btnClickTwo();
        } else if (id == R.id.dialog_txt_action_three) {
            this.onClickItemActionListener.btnClickThree();
        } else {
            int i = R.id.dialog_action_cancel;
        }
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_more);
        this.mTxtActionOne = (TextView) findViewById(R.id.dialog_txt_action_one);
        this.mTxtActionTwo = (TextView) findViewById(R.id.dialog_txt_action_two);
        this.mTxtActionThree = (TextView) findViewById(R.id.dialog_txt_action_three);
        this.mTxtCancel = (TextView) findViewById(R.id.dialog_action_cancel);
        this.mDeleteLine = findViewById(R.id.topic_action_delete_line);
        this.mTxtActionOne.setText(this.mTxtOne);
        this.mTxtActionTwo.setText(this.mTxtTwo);
        this.mTxtActionThree.setText(this.mTxtThree);
        LanguageUtil.setText(this.mActivity, this.mTxtCancel, "cancel");
        this.mTxtActionOne.setOnClickListener(this);
        this.mTxtActionTwo.setOnClickListener(this);
        this.mTxtActionThree.setOnClickListener(this);
        this.mTxtCancel.setOnClickListener(this);
        setViewLocation();
        setCanceledOnTouchOutside(true);
    }

    public void setOnClickItemActionListener(OnClickItemActionListener onClickItemActionListener) {
        this.onClickItemActionListener = onClickItemActionListener;
    }
}
