package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class ResetEmailActivity extends BaseActivity implements View.OnClickListener, TextWatcher {
    private RetrofitRequest baseRequest;
    private Button mBtnComplete;
    private Button mBtnGetCode;
    private EditText mEditNewEmailValue;
    private EditText mEditVerificationCode;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtEmail;
    private TextView mTxtEmailValue;
    private TextView mTxtNewEmail;
    private TextView mTxtVerificationCode;
    private int seconds = 90;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.ResetEmailActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 0) {
                int i = ResetEmailActivity.this.seconds;
                ResetEmailActivity resetEmailActivity = ResetEmailActivity.this;
                if (i > 0) {
                    resetEmailActivity.mBtnGetCode.setText(ResetEmailActivity.this.getResources().getString(R.string.android_hasSent) + "(" + ResetEmailActivity.this.seconds + "s)");
                    ResetEmailActivity.access$010(ResetEmailActivity.this);
                    ResetEmailActivity.this.mHandler.sendEmptyMessageDelayed(0, 1000L);
                } else {
                    resetEmailActivity.mBtnGetCode.setEnabled(true);
                    ResetEmailActivity.this.mBtnGetCode.setOnClickListener(ResetEmailActivity.this);
                    ResetEmailActivity.this.mBtnGetCode.setBackgroundResource(R.drawable.btn_get_code);
                    ResetEmailActivity.this.mBtnGetCode.setTextColor(ResetEmailActivity.this.getResources().getColor(R.color.get_code_text_color));
                    ResetEmailActivity resetEmailActivity2 = ResetEmailActivity.this;
                    LanguageUtil.setText(resetEmailActivity2, resetEmailActivity2.mBtnGetCode, "getVerificationCode");
                }
            }
            return false;
        }
    });

    static /* synthetic */ int access$010(ResetEmailActivity resetEmailActivity) {
        int i = resetEmailActivity.seconds;
        resetEmailActivity.seconds = i - 1;
        return i;
    }

    private void changeMail() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        final String trim = this.mEditNewEmailValue.getText().toString().trim();
        String obj = this.mEditVerificationCode.getText().toString();
        try {
            this.baseRequest.getApi().changeMail(str, trim, Integer.valueOf(!TextUtils.isEmpty(obj) ? Integer.valueOf(obj).intValue() : 0)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.ResetEmailActivity.3
                @Override // com.hismith.hismithapp.http.RxSubscribe
                protected void onSuccess(HttpModel httpModel) {
                    if (httpModel.getSuccess() == 1) {
                        SharedPreferencesUtil.put(ResetEmailActivity.this, "email", trim);
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setEmail(trim);
                            ResetEmailActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                        }
                        ResetEmailActivity.this.setResult(0);
                        ResetEmailActivity.this.finish();
                    }
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void checkSubmit() {
        String trim = this.mEditNewEmailValue.getText().toString().trim();
        String trim2 = this.mEditVerificationCode.getText().toString().trim();
        if (TextUtils.isEmpty(trim) || TextUtils.isEmpty(trim2)) {
            this.mBtnComplete.setEnabled(false);
            this.mBtnComplete.setBackground(ContextCompat.getDrawable(this, R.drawable.btn_main_submit30));
            this.mBtnComplete.setTextColor(getResources().getColor(R.color.white40));
        } else {
            this.mBtnComplete.setEnabled(true);
            this.mBtnComplete.setBackground(ContextCompat.getDrawable(this, R.drawable.btn_main_submit));
            this.mBtnComplete.setTextColor(getResources().getColor(R.color.white));
        }
    }

    private void getEmailVerCode() {
        this.baseRequest.getApi().emailVerCode((String) SharedPreferencesUtil.get(this, Constans.Token), "changemail", this.mEditNewEmailValue.getText().toString().trim()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.ResetEmailActivity.2
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ResetEmailActivity.this.mBtnGetCode.setEnabled(false);
                    ResetEmailActivity.this.mBtnGetCode.setBackgroundResource(R.drawable.btn_get_code30);
                    ResetEmailActivity.this.mBtnGetCode.setTextColor(ResetEmailActivity.this.getResources().getColor(R.color.get_code_text_color_disable));
                    ResetEmailActivity.this.seconds = 90;
                    ResetEmailActivity.this.mHandler.sendEmptyMessage(0);
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        checkSubmit();
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTxtEmail = (TextView) findViewById(R.id.reset_email_txt_email);
        this.mTxtEmailValue = (TextView) findViewById(R.id.reset_email_txt_emailValue);
        this.mTxtNewEmail = (TextView) findViewById(R.id.reset_email_txt_new_email);
        this.mEditNewEmailValue = (EditText) findViewById(R.id.reset_email_edit_new_emailValue);
        this.mTxtVerificationCode = (TextView) findViewById(R.id.reset_email_txt_verificationCode);
        this.mEditVerificationCode = (EditText) findViewById(R.id.reset_email_edit_verificationCode);
        this.mBtnGetCode = (Button) findViewById(R.id.reset_email_btn_getcode);
        this.mBtnComplete = (Button) findViewById(R.id.reset_email_btn_complete);
        LanguageUtil.setText(this, this.mTitleTxtName, "changeEmail");
        LanguageUtil.setText(this, this.mTxtEmail, "originalEmail");
        LanguageUtil.setText(this, this.mTxtNewEmail, "newEmail");
        LanguageUtil.setHint(this, this.mEditNewEmailValue, "pleaseEnterNewEmail");
        LanguageUtil.setText(this, this.mTxtVerificationCode, "verificationCode");
        LanguageUtil.setHint(this, this.mEditVerificationCode, "pleasseEnterVerificationCode");
        LanguageUtil.setText(this, this.mBtnGetCode, "getVerificationCode");
        LanguageUtil.setText(this, this.mBtnComplete, "confirm");
        String str = (String) SharedPreferencesUtil.get(this, "email");
        if (MyApplication.getApplication() != null && MyApplication.getApplication().getmUserModel() != null) {
            str = MyApplication.getApplication().getmUserModel().getEmail();
        }
        this.mTxtEmailValue.setText(str);
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mBtnComplete.setOnClickListener(this);
        this.mBtnGetCode.setOnClickListener(this);
        this.mEditNewEmailValue.addTextChangedListener(this);
        this.mEditVerificationCode.addTextChangedListener(this);
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
        } else if (id == R.id.reset_email_btn_getcode) {
            getEmailVerCode();
        } else if (id == R.id.reset_email_btn_complete) {
            changeMail();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_reset_email);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        this.mHandler.removeCallbacksAndMessages(null);
    }

    @Override // android.text.TextWatcher
    public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
    }
}
