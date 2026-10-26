package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;

/* loaded from: classes3.dex */
public abstract class RenameDialog extends Dialog implements View.OnClickListener {
    private Activity mActivity;
    private EditText mEditName;
    private String mName;
    private TextView mTxtCancel;
    private TextView mTxtConfirm;
    private TextView mTxtTitle;

    public RenameDialog(Activity activity, String str) {
        super(activity);
        this.mActivity = activity;
        this.mName = str;
    }

    public abstract void btnSubmit(String str);

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.rename_txt_cancel) {
            dismiss();
            return;
        }
        if (id == R.id.rename_txt_confirm) {
            String obj = this.mEditName.getText().toString();
            if (TextUtils.isEmpty(obj)) {
                return;
            }
            btnSubmit(obj);
            dismiss();
        }
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_rename);
        Window window = getWindow();
        WindowManager.LayoutParams attributes = window.getAttributes();
        attributes.width = (int) (this.mActivity.getWindowManager().getDefaultDisplay().getWidth() * 0.8d);
        window.setAttributes(attributes);
        this.mTxtTitle = (TextView) findViewById(R.id.rename_txt_title);
        this.mEditName = (EditText) findViewById(R.id.rename_edit_name);
        this.mTxtCancel = (TextView) findViewById(R.id.rename_txt_cancel);
        this.mTxtConfirm = (TextView) findViewById(R.id.rename_txt_confirm);
        LanguageUtil.setText(this.mActivity, this.mTxtTitle, "ModeName");
        LanguageUtil.setText(this.mActivity, this.mTxtCancel, "cancel");
        LanguageUtil.setText(this.mActivity, this.mTxtConfirm, "confirm");
        this.mEditName.setText(this.mName);
        this.mTxtCancel.setOnClickListener(this);
        this.mTxtConfirm.setOnClickListener(this);
        setCanceledOnTouchOutside(true);
    }
}
