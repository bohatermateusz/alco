package com.hismith.hismithapp.dialog;

import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.WheelView;

/* loaded from: classes3.dex */
public class GenderDialog extends Dialog implements View.OnClickListener {
    private Button mBtnCancel;
    private Button mBtnConfirm;
    private Context mContext;
    private TextView mTxtTitle;
    private WheelView mWheel;

    public interface OnSelectedListener {
        void onSelected(int i);
    }

    protected GenderDialog(Context context) {
        super(context);
        this.mContext = context;
    }

    protected GenderDialog(Context context, int i) {
        super(context, i);
        this.mContext = context;
    }

    protected GenderDialog(Context context, boolean z, DialogInterface.OnCancelListener onCancelListener) {
        super(context, z, onCancelListener);
        this.mContext = context;
    }

    private void init() {
    }

    private void setListener() {
        this.mBtnCancel.setOnClickListener(this);
        this.mBtnConfirm.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.dialog_gender_btn_cancel) {
            dismiss();
        } else {
            int i = R.id.dialog_gender_btn_confirm;
        }
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_select_gender);
        this.mBtnCancel = (Button) findViewById(R.id.dialog_gender_btn_cancel);
        this.mBtnConfirm = (Button) findViewById(R.id.dialog_gender_btn_confirm);
        this.mTxtTitle = (TextView) findViewById(R.id.dialog_gender_txt_title);
        this.mWheel = (WheelView) findViewById(R.id.dialog_gender_wheel);
        LanguageUtil.setText(this.mContext, this.mTxtTitle, "pleaseSelectDate");
        init();
        setListener();
    }
}
