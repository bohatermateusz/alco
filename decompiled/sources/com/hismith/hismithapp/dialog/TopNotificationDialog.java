package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class TopNotificationDialog extends Dialog implements View.OnClickListener {
    private String contentKey;
    private TextView dialog_txt_content;
    private TextView dialog_txt_title;
    private Activity mActivity;
    private String titleKey;

    public TopNotificationDialog(Activity activity, String str, String str2) {
        super(activity, R.style.ActionTopSheetDialogStyle);
        this.mActivity = activity;
        this.titleKey = str;
        this.contentKey = str2;
    }

    private void setViewLocation() {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        this.mActivity.getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int i = displayMetrics.heightPixels;
        WindowManager.LayoutParams attributes = getWindow().getAttributes();
        attributes.x = 0;
        attributes.y = 0;
        attributes.width = -1;
        attributes.height = -2;
        onWindowAttributesChanged(attributes);
    }

    private void setViewWidth() {
        Window window = getWindow();
        if (window != null) {
            window.setBackgroundDrawable(null);
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(this.mActivity).x * 0.5d);
            attributes.gravity = 48;
            window.setAttributes(attributes);
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        view.getId();
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_notification);
        setViewWidth();
        this.dialog_txt_title = (TextView) findViewById(R.id.dialog_txt_title);
        this.dialog_txt_content = (TextView) findViewById(R.id.dialog_txt_content);
        this.dialog_txt_title.setText(String.format(LanguageUtil.getLanguageByKey(this.mActivity, "permissionTitle"), LanguageUtil.getLanguageByKey(this.mActivity, this.titleKey)));
        LanguageUtil.setText(this.mActivity, this.dialog_txt_content, this.contentKey);
        setViewLocation();
        setCanceledOnTouchOutside(true);
    }
}
