package com.hismith.hismithapp;

import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.text.Editable;
import android.text.SpannableString;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.span.CenterAlignImageSpan;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class ResetActivity extends BaseActivity implements View.OnClickListener, TextWatcher {
    private RetrofitRequest baseRequest;
    private Button mBtnComplete;
    private EditText mEditConfirmPassword;
    private EditText mEditNewPassword;
    private EditText mEditOriginalPassword;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtConfirmPassword;
    private TextView mTxtEmail;
    private TextView mTxtEmailValue;
    private TextView mTxtLabel;
    private TextView mTxtNewPassword;
    private TextView mTxtOriginalPassword;

    private void changePass() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String trim = this.mEditOriginalPassword.getText().toString().trim();
        String trim2 = this.mEditNewPassword.getText().toString().trim();
        if (trim2.equals(this.mEditConfirmPassword.getText().toString().trim())) {
            this.baseRequest.getApi().changePass(str, trim, trim2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.ResetActivity.1
                @Override // com.hismith.hismithapp.http.RxSubscribe
                protected void onSuccess(HttpModel httpModel) {
                    if (httpModel.getSuccess() == 1) {
                        ToastUtil.showShortToastCenter(ResetActivity.this, R.string.android_updatedPasswordSuccess);
                        ResetActivity.this.finish();
                    }
                }
            });
        } else {
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "pleaseConfirmPassword"));
        }
    }

    private void checkSubmit() {
        String trim = this.mTxtEmailValue.getText().toString().trim();
        String trim2 = this.mEditOriginalPassword.getText().toString().trim();
        String trim3 = this.mEditNewPassword.getText().toString().trim();
        String trim4 = this.mEditConfirmPassword.getText().toString().trim();
        if (TextUtils.isEmpty(trim) || TextUtils.isEmpty(trim2) || TextUtils.isEmpty(trim3) || TextUtils.isEmpty(trim4)) {
            this.mBtnComplete.setEnabled(false);
            this.mBtnComplete.setBackgroundResource(R.drawable.btn_main_submit30);
            this.mBtnComplete.setTextColor(getResources().getColor(R.color.btn_main_submit30_text_color));
        } else {
            this.mBtnComplete.setEnabled(true);
            this.mBtnComplete.setBackgroundResource(R.drawable.btn_main_submit);
            this.mBtnComplete.setTextColor(getResources().getColor(R.color.btn_main_submit_text_color));
        }
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.mTxtEmailValue.setText((String) SharedPreferencesUtil.get(this, "email"));
        this.mTxtEmail.setText(this.mTxtEmail.getText().toString() + ":");
        this.mTxtOriginalPassword.setText(this.mTxtOriginalPassword.getText().toString() + ":");
        this.mTxtNewPassword.setText(this.mTxtNewPassword.getText().toString() + ":");
        this.mTxtConfirmPassword.setText(this.mTxtConfirmPassword.getText().toString() + ":");
        checkSubmit();
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTxtLabel = (TextView) findViewById(R.id.reset_txt_label);
        this.mTxtEmail = (TextView) findViewById(R.id.reset_txt_email);
        this.mTxtEmailValue = (TextView) findViewById(R.id.reset_txt_emailValue);
        this.mTxtOriginalPassword = (TextView) findViewById(R.id.reset_txt_originalPassword);
        this.mTxtNewPassword = (TextView) findViewById(R.id.reset_txt_NewPassword);
        this.mTxtConfirmPassword = (TextView) findViewById(R.id.reset_txt_confirmPassword);
        this.mEditOriginalPassword = (EditText) findViewById(R.id.reset_edit_originalPassword);
        this.mEditNewPassword = (EditText) findViewById(R.id.reset_edit_NewPassword);
        this.mEditConfirmPassword = (EditText) findViewById(R.id.reset_edit_confirmPassword);
        this.mBtnComplete = (Button) findViewById(R.id.reset_btn_complete);
        LanguageUtil.setText(this, this.mTitleTxtName, "resetPassword");
        LanguageUtil.setText(this, this.mTxtEmail, "email");
        LanguageUtil.setText(this, this.mTxtOriginalPassword, "originalPassword");
        LanguageUtil.setText(this, this.mTxtNewPassword, "pleaseEnterNewPassword");
        LanguageUtil.setText(this, this.mTxtConfirmPassword, "confirmPassword");
        LanguageUtil.setHint(this, this.mEditOriginalPassword, "pleaseEnterOriginalPassword");
        LanguageUtil.setHint(this, this.mEditNewPassword, "pleaseEnterNewPassword");
        LanguageUtil.setHint(this, this.mEditConfirmPassword, "pleasseEnterNewPasswordAgain");
        LanguageUtil.setText(this, this.mBtnComplete, "submit");
        SpannableString spannableString = new SpannableString("$  " + LanguageUtil.getLanguageByKey(this, "passwordLengthMinNote"));
        Drawable drawable = ContextCompat.getDrawable(this, R.mipmap.ic_tip);
        drawable.setBounds(0, 0, this.mTxtLabel.getLineHeight(), this.mTxtLabel.getLineHeight());
        spannableString.setSpan(new CenterAlignImageSpan(drawable), 0, 1, 1);
        this.mTxtLabel.setText(spannableString);
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mBtnComplete.setOnClickListener(this);
        this.mEditOriginalPassword.addTextChangedListener(this);
        this.mEditNewPassword.addTextChangedListener(this);
        this.mEditConfirmPassword.addTextChangedListener(this);
    }

    @Override // android.text.TextWatcher
    public void afterTextChanged(Editable editable) {
        checkSubmit();
    }

    @Override // android.text.TextWatcher
    public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
        } else if (id == R.id.reset_btn_complete) {
            changePass();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_reset);
        initView();
        init();
        setListener();
    }

    @Override // android.text.TextWatcher
    public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
    }
}
