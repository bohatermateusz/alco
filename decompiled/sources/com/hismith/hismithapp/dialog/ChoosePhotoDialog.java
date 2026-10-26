package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import com.facebook.internal.AnalyticsEvents;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;

/* loaded from: classes3.dex */
public abstract class ChoosePhotoDialog extends Dialog implements View.OnClickListener {
    private boolean isHideDelete;
    private Activity mActivity;
    private TextView mTxtCancel;
    private TextView mTxtDelete;
    private TextView mTxtPhoto;
    private TextView mTxtShoot;

    public ChoosePhotoDialog(Activity activity) {
        super(activity, R.style.ActionSheetDialogStyle);
        this.isHideDelete = true;
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

    public abstract void btnClickDelete();

    public abstract void btnClickPhoto();

    public abstract void btnClickShoot();

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.pic_txt_shoot) {
            btnClickShoot();
        } else if (id == R.id.pic_txt_photo) {
            btnClickPhoto();
        } else if (id == R.id.pic_txt_delete) {
            btnClickDelete();
        } else {
            int i = R.id.pic_txt_cancel;
        }
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_select_pic);
        this.mTxtShoot = (TextView) findViewById(R.id.pic_txt_shoot);
        this.mTxtPhoto = (TextView) findViewById(R.id.pic_txt_photo);
        this.mTxtDelete = (TextView) findViewById(R.id.pic_txt_delete);
        this.mTxtCancel = (TextView) findViewById(R.id.pic_txt_cancel);
        LanguageUtil.setText(this.mActivity, this.mTxtShoot, "shoot");
        LanguageUtil.setText(this.mActivity, this.mTxtPhoto, AnalyticsEvents.PARAMETER_SHARE_DIALOG_CONTENT_PHOTO);
        LanguageUtil.setText(this.mActivity, this.mTxtDelete, "delete");
        LanguageUtil.setText(this.mActivity, this.mTxtCancel, "cancel");
        this.mTxtShoot.setOnClickListener(this);
        this.mTxtPhoto.setOnClickListener(this);
        this.mTxtDelete.setOnClickListener(this);
        this.mTxtCancel.setOnClickListener(this);
        boolean z = this.isHideDelete;
        TextView textView = this.mTxtDelete;
        if (z) {
            textView.setVisibility(8);
        } else {
            textView.setVisibility(0);
        }
        setViewLocation();
        setCanceledOnTouchOutside(true);
    }

    public void setHideDelete(boolean z) {
        this.isHideDelete = z;
    }
}
