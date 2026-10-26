package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Editable;
import android.text.SpannableStringBuilder;
import android.text.TextPaint;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.LoginModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class RegisterActivity extends BaseActivity implements View.OnClickListener, TextWatcher {
    private RetrofitRequest baseRequest;
    private String email;
    private Button mBtnGetcode;
    private Button mBtnRegister;
    private CheckBox mCheckTerms;
    private EditText mEditCode;
    private EditText mEditMail;
    private EditText mEditPwd;
    private EditText mEditRepwd;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtCode;
    private TextView mTxtEmail;
    private TextView mTxtPwd;
    private TextView mTxtRePwd;
    private TextView mTxtTerms;
    private int seconds = 90;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.RegisterActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 0) {
                int i = RegisterActivity.this.seconds;
                RegisterActivity registerActivity = RegisterActivity.this;
                if (i > 0) {
                    RegisterActivity.this.mBtnGetcode.setText(LanguageUtil.getLanguageByKey(registerActivity, "hasSent") + " (" + RegisterActivity.this.seconds + "s)");
                    RegisterActivity.access$010(RegisterActivity.this);
                    RegisterActivity.this.mHandler.sendEmptyMessageDelayed(0, 1000L);
                } else {
                    registerActivity.mBtnGetcode.setEnabled(true);
                    RegisterActivity.this.mBtnGetcode.setOnClickListener(RegisterActivity.this);
                    RegisterActivity registerActivity2 = RegisterActivity.this;
                    LanguageUtil.setText(registerActivity2, registerActivity2.mBtnGetcode, "getVerificationCode");
                }
            }
            return false;
        }
    });

    static /* synthetic */ int access$010(RegisterActivity registerActivity) {
        int i = registerActivity.seconds;
        registerActivity.seconds = i - 1;
        return i;
    }

    private void callService(String str, final TextView textView) {
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder(str);
        spannableStringBuilder.setSpan(new ClickableSpan() { // from class: com.hismith.hismithapp.RegisterActivity.3
            @Override // android.text.style.ClickableSpan
            public void onClick(View view) {
                RegisterActivity.this.startActivity(new Intent(RegisterActivity.this, (Class<?>) UserTermsActivity.class));
            }

            @Override // android.text.style.ClickableSpan, android.text.style.CharacterStyle
            public void updateDrawState(TextPaint textPaint) {
                super.updateDrawState(textPaint);
                textPaint.setColor(textView.getCurrentTextColor());
                textPaint.setUnderlineText(true);
            }
        }, 0, str.length(), 33);
        textView.setHighlightColor(0);
        textView.setText(spannableStringBuilder);
        textView.setMovementMethod(LinkMovementMethod.getInstance());
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void checkSubmit() {
        String trim = this.mEditMail.getText().toString().trim();
        String trim2 = this.mEditCode.getText().toString().trim();
        String trim3 = this.mEditPwd.getText().toString().trim();
        String trim4 = this.mEditRepwd.getText().toString().trim();
        boolean isChecked = this.mCheckTerms.isChecked();
        if (TextUtils.isEmpty(trim) || TextUtils.isEmpty(trim2) || TextUtils.isEmpty(trim3) || TextUtils.isEmpty(trim4) || !isChecked) {
            this.mBtnRegister.setEnabled(false);
            this.mBtnRegister.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_style_25));
            this.mBtnRegister.setTextColor(ContextCompat.getColor(this, R.color.white40));
        } else {
            this.mBtnRegister.setEnabled(true);
            this.mBtnRegister.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_style_25));
            this.mBtnRegister.setTextColor(ContextCompat.getColor(this, R.color.white));
        }
    }

    private void getEmailVerCode() {
        this.baseRequest.getApi().emailVerCode("register", this.mEditMail.getText().toString().trim()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.RegisterActivity.4
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    RegisterActivity.this.mBtnGetcode.setEnabled(false);
                    RegisterActivity.this.seconds = 90;
                    RegisterActivity.this.mHandler.sendEmptyMessage(0);
                }
            }
        });
    }

    private void initValue() {
        this.baseRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_BASE);
        checkSubmit();
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTxtEmail = (TextView) findViewById(R.id.register_txt_mail);
        this.mTxtCode = (TextView) findViewById(R.id.register_txt_code);
        this.mTxtPwd = (TextView) findViewById(R.id.register_txt_pwd);
        this.mTxtRePwd = (TextView) findViewById(R.id.register_txt_repwd);
        this.mCheckTerms = (CheckBox) findViewById(R.id.register_check_terms);
        this.mTxtTerms = (TextView) findViewById(R.id.register_txt_terms);
        this.mEditMail = (EditText) findViewById(R.id.register_edit_mail);
        this.mEditCode = (EditText) findViewById(R.id.register_edit_code);
        this.mEditPwd = (EditText) findViewById(R.id.register_edit_pwd);
        this.mEditRepwd = (EditText) findViewById(R.id.register_edit_repwd);
        this.mBtnGetcode = (Button) findViewById(R.id.register_btn_getcode);
        this.mBtnRegister = (Button) findViewById(R.id.register_btn_register);
        LanguageUtil.setText(this, this.mTxtEmail, "email");
        LanguageUtil.setText(this, this.mTxtCode, "pleasseEnterVerificationCode");
        LanguageUtil.setText(this, this.mTxtPwd, "password");
        LanguageUtil.setText(this, this.mTxtRePwd, "password");
        LanguageUtil.setText(this, this.mTitleTxtName, "register");
        LanguageUtil.setText(this, this.mBtnGetcode, "getVerificationCode");
        LanguageUtil.setText(this, this.mBtnRegister, "registerNow");
        LanguageUtil.setHint(this, this.mEditMail, "pleaseEnterEmail");
        LanguageUtil.setHint(this, this.mEditCode, "pleasseEnterVerificationCode");
        LanguageUtil.setHint(this, this.mEditPwd, "pleasseEnterPassword");
        LanguageUtil.setHint(this, this.mEditRepwd, "pleaseEnterPasswordAgain");
        callService(LanguageUtil.getLanguageByKey(this, "userTerms"), this.mTxtTerms);
    }

    private void register() {
        this.email = this.mEditMail.getText().toString().trim();
        String obj = this.mEditPwd.getText().toString();
        String obj2 = this.mEditRepwd.getText().toString();
        String obj3 = this.mEditCode.getText().toString();
        if (!obj.equals(obj2)) {
            ToastUtil.showShortToastCenter(this, R.string.android_pleaseConfirmPassword);
            return;
        }
        try {
            this.baseRequest.getApi().register(this.email, obj, TextUtils.isEmpty(obj3) ? 0 : Integer.valueOf(obj3)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LoginModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.RegisterActivity.5
                /* JADX INFO: Access modifiers changed from: protected */
                @Override // com.hismith.hismithapp.http.RxSubscribe
                public void onSuccess(LoginModel loginModel) {
                    if (loginModel.getSuccess() == 1) {
                        ToastUtil.showShortToastCenter(RegisterActivity.this, R.string.android_registrationSuccess);
                        Intent intent = new Intent();
                        intent.putExtra(Constans.Token, loginModel.getToken());
                        intent.putExtra("email", RegisterActivity.this.email);
                        RegisterActivity.this.setResult(0, intent);
                        Intent intent2 = new Intent(RegisterActivity.this, (Class<?>) InfoActivity.class);
                        intent2.putExtra(Constans.Token, loginModel.getToken());
                        RegisterActivity.this.startActivity(intent2);
                        RegisterActivity.this.finish();
                    }
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mBtnGetcode.setOnClickListener(this);
        this.mBtnRegister.setOnClickListener(this);
        this.mEditMail.addTextChangedListener(this);
        this.mEditCode.addTextChangedListener(this);
        this.mEditPwd.addTextChangedListener(this);
        this.mEditRepwd.addTextChangedListener(this);
        this.mCheckTerms.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() { // from class: com.hismith.hismithapp.RegisterActivity.2
            @Override // android.widget.CompoundButton.OnCheckedChangeListener
            public void onCheckedChanged(CompoundButton compoundButton, boolean z) {
                RegisterActivity.this.checkSubmit();
            }
        });
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
            this.appManager.finishActivity();
        } else if (id == R.id.register_btn_getcode) {
            getEmailVerCode();
        } else if (id == R.id.register_btn_register) {
            register();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_register);
        initView();
        initValue();
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
