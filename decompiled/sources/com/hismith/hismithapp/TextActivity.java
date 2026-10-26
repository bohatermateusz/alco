package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import com.hismith.hismithapp.utils.LanguageUtil;

/* loaded from: classes3.dex */
public class TextActivity extends BaseActivity implements View.OnClickListener {
    public static final String EXTRA_TEXT_CONTENT = "extra_text_content";
    private String mContent;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtContent;

    private void init() {
        if (getIntent() != null) {
            String stringExtra = getIntent().getStringExtra(EXTRA_TEXT_CONTENT);
            this.mContent = stringExtra;
            this.mTxtContent.setText(stringExtra);
        }
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTxtContent = (TextView) findViewById(R.id.text_txt_content);
        LanguageUtil.setText(this, this.mTitleTxtName, "scanResult");
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            finish();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_text);
        initView();
        init();
        setListener();
    }
}
