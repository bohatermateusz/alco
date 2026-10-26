package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import com.hismith.hismithapp.utils.LanguageUtil;

/* loaded from: classes3.dex */
public class SettingActivity extends BaseActivity implements View.OnClickListener {
    private TextView mTitleName;
    private ImageView mTitltBack;

    private void init() {
    }

    private void initView() {
        this.mTitltBack = (ImageView) findViewById(R.id.title_img_back);
        TextView textView = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleName = textView;
        LanguageUtil.setText(this, textView, "setting");
    }

    private void setListener() {
        this.mTitltBack.setOnClickListener(this);
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
        setContentView(R.layout.activity_setting);
        initView();
        init();
        setListener();
    }
}
