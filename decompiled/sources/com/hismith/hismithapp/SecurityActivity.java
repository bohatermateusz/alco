package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import com.hismith.hismithapp.utils.LanguageUtil;

/* loaded from: classes3.dex */
public class SecurityActivity extends BaseActivity implements View.OnClickListener {
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtResetPassword;
    private TextView mTxtUnsubscribe;

    private void init() {
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTxtResetPassword = (TextView) findViewById(R.id.security_txt_resetPassword);
        this.mTxtUnsubscribe = (TextView) findViewById(R.id.security_txt_unsubscribe);
        LanguageUtil.setText(this, this.mTitleTxtName, "accountSecurity");
        LanguageUtil.setText(this, this.mTxtResetPassword, "resetPassword");
        LanguageUtil.setText(this, this.mTxtUnsubscribe, "accountCancellation");
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mTxtResetPassword.setOnClickListener(this);
        this.mTxtUnsubscribe.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
        } else if (id == R.id.security_txt_resetPassword) {
            startActivity(new Intent(this, (Class<?>) ResetActivity.class));
        } else if (id == R.id.security_txt_unsubscribe) {
            startActivity(new Intent(this, (Class<?>) UnsubscribeActivity.class));
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_security);
        initView();
        init();
        setListener();
    }
}
