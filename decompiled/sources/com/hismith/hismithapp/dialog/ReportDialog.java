package com.hismith.hismithapp.dialog;

import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public abstract class ReportDialog extends Dialog implements View.OnClickListener {
    private Button mBtnSubmit;
    private Context mContext;
    private EditText mEditReason;
    private TextView mTxtTitle;

    public ReportDialog(Context context) {
        super(context);
        this.mContext = context;
    }

    public abstract void btnSubmit(String str);

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.report_btn_submit) {
            btnSubmit(this.mEditReason.getText().toString());
        }
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_report);
        Window window = getWindow();
        WindowManager.LayoutParams attributes = window.getAttributes();
        attributes.width = (int) (MeasureUtil.getScreenWidth(this.mContext) * 0.8d);
        window.setAttributes(attributes);
        this.mTxtTitle = (TextView) findViewById(R.id.report_txt_title);
        this.mEditReason = (EditText) findViewById(R.id.report_edit_reason);
        this.mBtnSubmit = (Button) findViewById(R.id.report_btn_submit);
        LanguageUtil.setText(this.mContext, this.mTxtTitle, "reportReason");
        LanguageUtil.setText(this.mContext, this.mBtnSubmit, "submit");
        this.mBtnSubmit.setOnClickListener(this);
        setCanceledOnTouchOutside(true);
    }
}
