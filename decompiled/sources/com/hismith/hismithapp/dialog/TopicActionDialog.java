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
public class TopicActionDialog extends Dialog implements View.OnClickListener {
    private boolean isShow;
    private Activity mActivity;
    private View mDeleteLine;
    private TextView mTxtCancel;
    private TextView mTxtDelete;
    private TextView mTxtNotInterested;
    private TextView mTxtReport;
    private OnClickItemActionListener onClickItemActionListener;
    private int position;

    public interface OnClickItemActionListener {
        void btnClickDelete(int i);

        void btnClickNotInterested(int i);

        void btnClickReport(int i);
    }

    public TopicActionDialog(Activity activity) {
        super(activity, R.style.ActionSheetDialogStyle);
        this.isShow = true;
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

    public TopicActionDialog hideDelete() {
        this.isShow = false;
        return this;
    }

    public TopicActionDialog initDelete() {
        TextView textView = this.mTxtDelete;
        if (textView == null) {
            return this;
        }
        if (this.isShow) {
            textView.setVisibility(0);
            this.mDeleteLine.setVisibility(0);
            return this;
        }
        textView.setVisibility(8);
        this.mDeleteLine.setVisibility(8);
        return this;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.topic_action_report) {
            this.onClickItemActionListener.btnClickReport(this.position);
        } else if (id == R.id.topic_action_notInterested) {
            this.onClickItemActionListener.btnClickNotInterested(this.position);
        } else if (id == R.id.topic_action_delete) {
            this.onClickItemActionListener.btnClickDelete(this.position);
        } else {
            int i = R.id.topic_action_cancel;
        }
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_select_topic_action);
        this.mTxtReport = (TextView) findViewById(R.id.topic_action_report);
        this.mTxtNotInterested = (TextView) findViewById(R.id.topic_action_notInterested);
        this.mTxtDelete = (TextView) findViewById(R.id.topic_action_delete);
        this.mTxtCancel = (TextView) findViewById(R.id.topic_action_cancel);
        this.mDeleteLine = findViewById(R.id.topic_action_delete_line);
        LanguageUtil.setText(this.mActivity, this.mTxtReport, "report");
        LanguageUtil.setText(this.mActivity, this.mTxtNotInterested, "notInterested");
        LanguageUtil.setText(this.mActivity, this.mTxtDelete, "delete");
        LanguageUtil.setText(this.mActivity, this.mTxtCancel, "cancel");
        boolean z = this.isShow;
        TextView textView = this.mTxtDelete;
        if (z) {
            textView.setVisibility(0);
            this.mDeleteLine.setVisibility(0);
        } else {
            textView.setVisibility(8);
            this.mDeleteLine.setVisibility(8);
        }
        this.mTxtReport.setOnClickListener(this);
        this.mTxtNotInterested.setOnClickListener(this);
        this.mTxtDelete.setOnClickListener(this);
        this.mTxtCancel.setOnClickListener(this);
        setViewLocation();
        setCanceledOnTouchOutside(true);
    }

    public void setOnClickItemActionListener(OnClickItemActionListener onClickItemActionListener, int i) {
        this.onClickItemActionListener = onClickItemActionListener;
        this.position = i;
    }

    public TopicActionDialog showDelete() {
        this.isShow = true;
        return this;
    }
}
