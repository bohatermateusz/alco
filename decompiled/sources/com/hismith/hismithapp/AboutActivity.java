package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;

/* loaded from: classes3.dex */
public class AboutActivity extends BaseActivity implements View.OnClickListener {
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtAboutApp;
    private TextView mTxtAboutusPrivacyNotice;
    private TextView mTxtAboutusRateme;
    private TextView mTxtAboutusUserTerm;
    private TextView mTxtCopyright;
    private TextView mTxtName;
    private TextView mTxtVersion;

    private void init() {
        this.mTxtVersion.setText(AppUtil.getAppVersionName(this));
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTxtName = (TextView) findViewById(R.id.about_txt_name);
        this.mTxtVersion = (TextView) findViewById(R.id.about_txt_version);
        this.mTxtAboutApp = (TextView) findViewById(R.id.about_txt_aboutApp);
        this.mTxtAboutusRateme = (TextView) findViewById(R.id.about_txt_aboutusRateme);
        this.mTxtAboutusUserTerm = (TextView) findViewById(R.id.about_txt_aboutusUserTerm);
        this.mTxtAboutusPrivacyNotice = (TextView) findViewById(R.id.about_txt_aboutusPrivacyNotice);
        this.mTxtCopyright = (TextView) findViewById(R.id.about_txt_copyright);
        this.mTxtName.setText(AppUtil.getAppName(this));
        LanguageUtil.setText(this, this.mTitleTxtName, "aboutus");
        LanguageUtil.setText(this, this.mTxtAboutusRateme, "aboutusRateme");
        LanguageUtil.setText(this, this.mTxtAboutusUserTerm, "aboutusUserTerm");
        LanguageUtil.setText(this, this.mTxtAboutusPrivacyNotice, "aboutusPrivacyNotice");
        String str = (String) SharedPreferencesUtil.get(this, Constans.aboutApp);
        String str2 = (String) SharedPreferencesUtil.get(this, Constans.copyright);
        this.mTxtAboutApp.setText(str);
        this.mTxtCopyright.setText(str2);
        this.mTxtAboutusRateme.setVisibility(8);
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mTxtAboutusUserTerm.setOnClickListener(this);
        this.mTxtAboutusPrivacyNotice.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
        } else if (id == R.id.about_txt_aboutusUserTerm) {
            startActivity(new Intent(this, (Class<?>) UserTermsActivity.class));
        } else if (id == R.id.about_txt_aboutusPrivacyNotice) {
            startActivity(new Intent(this, (Class<?>) PrivacyNoticeActivity.class));
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_about);
        initView();
        init();
        setListener();
    }
}
