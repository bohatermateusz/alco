package com.hismith.hismithapp;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.os.Bundle;
import android.text.Editable;
import android.text.SpannableStringBuilder;
import android.text.TextPaint;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.facebook.AccessToken;
import com.facebook.FacebookException;
import com.facebook.login.LoginResult;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.GoogleApiAvailability;
import com.google.android.gms.common.api.GoogleApiClient;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.adapter.LanguageAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.FaceBookLoginManager;
import com.hismith.hismithapp.manager.GoogleLoginManager;
import com.hismith.hismithapp.manager.UpgradeManager;
import com.hismith.hismithapp.model.LoginModel;
import com.hismith.hismithapp.twitter.OAuthSigner;
import com.hismith.hismithapp.twitter.TwitterOAuth2Activity;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.twitter.sdk.android.core.identity.TwitterAuthClient;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/* loaded from: classes3.dex */
public class LoginActivity extends BaseActivity implements View.OnClickListener, GoogleApiClient.OnConnectionFailedListener, GoogleLoginManager.OnLoginSuccessListener, TextWatcher {
    private static final int requestRegister = 111;
    private RetrofitRequest baseRequest;
    private GoogleSignInOptions gso;
    private Button mBtnLogin;
    private EditText mEditMail;
    private EditText mEditPwd;
    private GoogleApiClient mGoogleApiClient;
    private ImageButton mImgbtnFacebook;
    private ImageButton mImgbtnGoogle;
    private ImageButton mImgbtnwitter;
    private LinearLayout mLayout;
    private LinearLayout mLayoutLanguage;
    private LinearLayoutManager mLayoutManager;
    private LinearLayout mLayoutOffLine;
    private RecyclerView mRecyclerView;
    private TextView mTextForgot;
    private TextView mTextRegister;
    private TwitterAuthClient mTwitterAuthClient;
    private TextView mTxtLanguage;
    private TextView mTxtOffLine;
    private TextView mTxtUserTerms;
    private OAuthSigner oAuthSigner;
    private LanguageAdapter popDataAdapter;
    private PopupWindow popupWindow;
    private String userName = "";
    private int requestGoogle = 222;
    private int requestTwitter = 333;

    private void authTwitter() {
        Intent intent = new Intent(this, (Class<?>) TwitterOAuth2Activity.class);
        intent.putExtra("clientId", "ZGdTajVVOUJXc1J1TzhaZF8wbzQ6MTpjaQ");
        startActivityForResult(intent, this.requestTwitter);
    }

    private float autosizeText(float f) {
        return f / (getResources().getDisplayMetrics().density + 0.2f);
    }

    private void callService(String str, final TextView textView) {
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder(str);
        spannableStringBuilder.setSpan(new ClickableSpan() { // from class: com.hismith.hismithapp.LoginActivity.1
            @Override // android.text.style.ClickableSpan
            public void onClick(View view) {
                if (view.getId() == R.id.login_txt_userTerms) {
                    LoginActivity.this.startActivity(new Intent(LoginActivity.this, (Class<?>) UserTermsActivity.class));
                }
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
    public void changeLanguage(String str) {
        SharedPreferencesUtil.put(this, Constans.Language, str);
        MyApplication.getApplication().initLanguage();
        recreate();
    }

    private void checkSubmit() {
        String trim = this.mEditMail.getText().toString().trim();
        String trim2 = this.mEditPwd.getText().toString().trim();
        if (TextUtils.isEmpty(trim) || TextUtils.isEmpty(trim2)) {
            this.mBtnLogin.setEnabled(false);
            this.mBtnLogin.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_style30_25));
            this.mBtnLogin.setTextColor(ContextCompat.getColor(this, R.color.white40));
        } else {
            this.mBtnLogin.setEnabled(true);
            this.mBtnLogin.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_style_25));
            this.mBtnLogin.setTextColor(ContextCompat.getColor(this, R.color.white));
        }
    }

    private void facebookLogin() {
        FaceBookLoginManager.getInstance().initFaceBook(this, new FaceBookLoginManager.OnLoginSuccessListener() { // from class: com.hismith.hismithapp.LoginActivity.4
            @Override // com.hismith.hismithapp.manager.FaceBookLoginManager.OnLoginSuccessListener
            public void onCancel() {
            }

            @Override // com.hismith.hismithapp.manager.FaceBookLoginManager.OnLoginSuccessListener
            public void onError(FacebookException facebookException) {
            }

            @Override // com.hismith.hismithapp.manager.FaceBookLoginManager.OnLoginSuccessListener
            public void onSuccess(LoginResult loginResult) {
                LoginActivity.this.otherLogin(AccessToken.DEFAULT_GRAPH_DOMAIN, loginResult.getAccessToken().getToken());
            }
        });
        FaceBookLoginManager.getInstance().faceBookLogin(this);
    }

    public static void getKeyHash(Context context, String str) {
        try {
            for (Signature signature : context.getPackageManager().getPackageInfo(str, 64).signatures) {
                MessageDigest messageDigest = MessageDigest.getInstance("SHA");
                messageDigest.update(signature.toByteArray());
                LogUtil.e(Base64.encodeToString(messageDigest.digest(), 0));
            }
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        } catch (NoSuchAlgorithmException e2) {
            e2.printStackTrace();
        }
    }

    private void googleLogin() {
        if (GoogleApiAvailability.getInstance().isGooglePlayServicesAvailable(this) == 0) {
            GoogleLoginManager.initGoogle(this, getResources().getString(R.string.google_client_id), this.requestGoogle, this.progressDialog);
        } else {
            GoogleApiAvailability.getInstance().makeGooglePlayServicesAvailable(this);
        }
    }

    private void initPop(final TextView textView) {
        RecyclerView recyclerView = new RecyclerView(this);
        this.mRecyclerView = recyclerView;
        recyclerView.setBackgroundColor(getResources().getColor(R.color.white));
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        LanguageAdapter languageAdapter = new LanguageAdapter(this, LanguageUtil.getLanguageList());
        this.popDataAdapter = languageAdapter;
        languageAdapter.setOnItemClickListener(new LanguageAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.LoginActivity.2
            @Override // com.hismith.hismithapp.adapter.LanguageAdapter.OnItemClickListener
            public void onItemClick(final int i) {
                textView.setText(LanguageUtil.getLanguageList().get(i).getName());
                LoginActivity.this.popupWindow.dismiss();
                Observable<HttpModel> observeOn = LoginActivity.this.baseRequest.getApi().setUserLang((String) SharedPreferencesUtil.get(LoginActivity.this, Constans.tempToken), LanguageUtil.getLanguageList().get(i).getIso()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
                LoginActivity loginActivity = LoginActivity.this;
                observeOn.subscribe(new RxSubscribe<HttpModel>(loginActivity, loginActivity.progressDialog) { // from class: com.hismith.hismithapp.LoginActivity.2.1
                    @Override // com.hismith.hismithapp.http.RxSubscribe
                    protected void onSuccess(HttpModel httpModel) {
                        if (httpModel.getSuccess() == 1) {
                            LoginActivity.this.changeLanguage(LanguageUtil.getLanguageList().get(i).getIso());
                        }
                    }
                });
            }
        });
        this.mRecyclerView.setAdapter(this.popDataAdapter);
        PopupWindow popupWindow = new PopupWindow((View) this.mRecyclerView, this.mLayoutLanguage.getWidth(), -2, true);
        this.popupWindow = popupWindow;
        popupWindow.setOnDismissListener(new PopupWindow.OnDismissListener() { // from class: com.hismith.hismithapp.LoginActivity.3
            @Override // android.widget.PopupWindow.OnDismissListener
            public void onDismiss() {
                LoginActivity.this.popupWindow.dismiss();
            }
        });
        this.popupWindow.setAnimationStyle(R.style.popmenu_animation);
        this.popupWindow.setFocusable(true);
        this.popupWindow.setOutsideTouchable(true);
    }

    private void initValue() {
        this.mLayout.measure(0, 0);
        this.mLayout.getMeasuredWidth();
        if (this.mLayout.getMeasuredHeight() < getResources().getDisplayMetrics().heightPixels) {
            FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) this.mLayout.getLayoutParams();
            layoutParams.gravity = 17;
            this.mLayout.setLayoutParams(layoutParams);
        }
        this.baseRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_BASE);
        this.gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN).requestEmail().build();
        this.mTxtLanguage.setText(LanguageUtil.getCurrLanguageName(LanguageUtil.getCurrLanguage(this)));
        LanguageUtil.setHint(this, this.mEditMail, "phoneEmailID");
        LanguageUtil.setHint(this, this.mEditPwd, "password");
        LanguageUtil.setText(this, this.mTextRegister, "register");
        LanguageUtil.setText(this, this.mTextForgot, "forgotPassword");
        LanguageUtil.setText(this, this.mBtnLogin, FirebaseAnalytics.Event.LOGIN);
        LanguageUtil.setText(this, this.mTxtOffLine, "playOffline");
        callService(LanguageUtil.getLanguageByKey(this, "userTerms"), this.mTxtUserTerms);
        this.mEditMail.setTextSize(autosizeText(this.mTextRegister.getTextSize()));
        this.mEditPwd.setTextSize(autosizeText(this.mTextRegister.getTextSize()));
        this.oAuthSigner = new OAuthSigner(this);
        checkSubmit();
    }

    private void initView() {
        this.mLayout = (LinearLayout) findViewById(R.id.login_layout);
        this.mLayoutLanguage = (LinearLayout) findViewById(R.id.login_layout_language);
        this.mEditMail = (EditText) findViewById(R.id.login_edit_mail);
        this.mEditPwd = (EditText) findViewById(R.id.login_edit_pwd);
        this.mTextRegister = (TextView) findViewById(R.id.login_txt_register);
        this.mTextForgot = (TextView) findViewById(R.id.login_txt_forgot);
        this.mTxtLanguage = (TextView) findViewById(R.id.login_txt_language);
        this.mBtnLogin = (Button) findViewById(R.id.login_btn_login);
        this.mImgbtnGoogle = (ImageButton) findViewById(R.id.login_imgbtn_google);
        this.mImgbtnwitter = (ImageButton) findViewById(R.id.login_imgbtn_twitter);
        this.mImgbtnFacebook = (ImageButton) findViewById(R.id.login_imgbtn_facebook);
        this.mTxtUserTerms = (TextView) findViewById(R.id.login_txt_userTerms);
        this.mLayoutOffLine = (LinearLayout) findViewById(R.id.login_layout_off_line);
        this.mTxtOffLine = (TextView) findViewById(R.id.login_txt_off_line);
    }

    private void login() {
        this.userName = this.mEditMail.getText().toString().trim();
        this.baseRequest.getApi().login(this.userName, this.mEditPwd.getText().toString(), "app").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LoginModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.LoginActivity.6
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(LoginModel loginModel) {
                if (loginModel.getSuccess() == 1) {
                    if (!TextUtils.isEmpty(loginModel.getUserLangIso())) {
                        SharedPreferencesUtil.put(LoginActivity.this, Constans.Language, loginModel.getUserLangIso());
                        MyApplication.getApplication().initLanguage();
                    }
                    Intent intent = new Intent(LoginActivity.this, (Class<?>) LoginTermsActivity.class);
                    intent.putExtra(Constans.Token, loginModel.getToken());
                    intent.putExtra("email", LoginActivity.this.userName);
                    LoginActivity.this.startActivity(intent);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void otherLogin(String str, String str2) {
        this.baseRequest.getApi().userOauthLogin(str, str2, "1.0").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LoginModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.LoginActivity.7
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(LoginModel loginModel) {
                if (loginModel.getSuccess() == 1) {
                    if (!TextUtils.isEmpty(loginModel.getUserLangIso())) {
                        SharedPreferencesUtil.put(LoginActivity.this, Constans.Language, loginModel.getUserLangIso());
                        MyApplication.getApplication().initLanguage();
                    }
                    Intent intent = new Intent(LoginActivity.this, (Class<?>) LoginTermsActivity.class);
                    intent.putExtra(Constans.Token, loginModel.getToken());
                    intent.putExtra("email", "");
                    LoginActivity.this.startActivity(intent);
                }
            }
        });
    }

    public static int px2dip(Context context, float f) {
        return (int) ((f / context.getResources().getDisplayMetrics().density) + 0.5f);
    }

    private void setListener() {
        this.mTextRegister.setOnClickListener(this);
        this.mTextForgot.setOnClickListener(this);
        this.mLayoutLanguage.setOnClickListener(this);
        this.mBtnLogin.setOnClickListener(this);
        this.mImgbtnGoogle.setOnClickListener(this);
        this.mImgbtnwitter.setOnClickListener(this);
        this.mImgbtnFacebook.setOnClickListener(this);
        this.mLayoutOffLine.setOnClickListener(this);
        this.mEditMail.addTextChangedListener(this);
        this.mEditPwd.addTextChangedListener(this);
    }

    private void showError(String str) {
        LogUtil.e("twitter error :" + str);
        ToastUtil.showLongToast(this, str);
    }

    @Override // android.text.TextWatcher
    public void afterTextChanged(Editable editable) {
        this.mEditMail.setTextSize(autosizeText(this.mTextRegister.getTextSize()));
        this.mEditPwd.setTextSize(autosizeText(this.mTextRegister.getTextSize()));
        checkSubmit();
    }

    @Override // android.text.TextWatcher
    public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        FaceBookLoginManager.getInstance().setOnActivityResult(i, i2, intent);
        TwitterAuthClient twitterAuthClient = this.mTwitterAuthClient;
        if (twitterAuthClient != null) {
            twitterAuthClient.onActivityResult(i, i2, intent);
        }
        GoogleLoginManager.onActivityResult(i, intent, this.requestGoogle, this, this.progressDialog);
        if (i == this.requestTwitter) {
            if (intent != null) {
                String stringExtra = intent.getStringExtra("code");
                String stringExtra2 = intent.getStringExtra("challenge");
                LogUtil.e("twitter oauth2 code:" + stringExtra + ", challenge:" + stringExtra2);
                this.oAuthSigner.getOauth2AccessToken2(stringExtra, stringExtra2, new OAuthSigner.OnOtherLoginListener() { // from class: com.hismith.hismithapp.LoginActivity.5
                    @Override // com.hismith.hismithapp.twitter.OAuthSigner.OnOtherLoginListener
                    public void onOtherLogin(final String str) {
                        LoginActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.LoginActivity.5.1
                            @Override // java.lang.Runnable
                            public void run() {
                                LoginActivity.this.otherLogin("twitter", str);
                            }
                        });
                    }
                });
            } else {
                LogUtil.e("twitter onActivityResult data：null");
            }
        }
        if (i != requestRegister || intent == null) {
            return;
        }
        String stringExtra3 = intent.getStringExtra(Constans.Token);
        String stringExtra4 = intent.getStringExtra("email");
        if (TextUtils.isEmpty(stringExtra3)) {
            return;
        }
        Intent intent2 = new Intent(this, (Class<?>) LoginTermsActivity.class);
        intent2.putExtra(Constans.Token, stringExtra3);
        intent2.putExtra("email", stringExtra4);
        startActivity(intent2);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.login_txt_register) {
            startActivityForResult(new Intent(this, (Class<?>) RegisterActivity.class), requestRegister);
            return;
        }
        if (id == R.id.login_txt_forgot) {
            startActivityForResult(new Intent(this, (Class<?>) ForgotActivity.class), requestRegister);
            return;
        }
        if (id == R.id.login_layout_language) {
            if (this.popupWindow == null) {
                initPop(this.mTxtLanguage);
            }
            if (this.popupWindow.isShowing()) {
                return;
            }
            this.popupWindow.showAsDropDown(this.mLayoutLanguage, 0, MeasureUtil.dip2px(this, 5.0f));
            return;
        }
        if (id == R.id.login_btn_login) {
            login();
            return;
        }
        if (id == R.id.login_imgbtn_google) {
            googleLogin();
            return;
        }
        if (id == R.id.login_imgbtn_twitter) {
            authTwitter();
        } else if (id != R.id.login_imgbtn_facebook && id == R.id.login_layout_off_line) {
            startActivity(new Intent(this, (Class<?>) IPlayOffActivity.class));
        }
    }

    @Override // com.google.android.gms.common.api.internal.OnConnectionFailedListener
    public void onConnectionFailed(ConnectionResult connectionResult) {
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_login);
        initView();
        initValue();
        setListener();
        UpgradeManager.initUpgrade(this);
    }

    @Override // com.hismith.hismithapp.manager.GoogleLoginManager.OnLoginSuccessListener
    public void onSuccessResult(String str) {
        otherLogin("google", str);
    }

    @Override // android.text.TextWatcher
    public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
    }
}
