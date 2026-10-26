package com.hismith.hismithapp;

import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Editable;
import android.text.SpannableString;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.span.CenterAlignImageSpan;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class UnsubscribeActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbWrite;
    private Button mBtnCode;
    private Button mBtnError;
    private Button mBtnSubmit;
    private EditText mEditCode;
    private TextView mEditEmail;
    private ImageView mImgCode;
    private ImageView mImgEmail;
    private ImageView mImgError;
    private LinearLayout mLayoutEdit;
    private LinearLayout mLayoutEmpty;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtError;
    private TextView mTxtRemoveAccountNote;
    private int seconds = 90;
    private final int REQUEST_EMAIL = 222;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.UnsubscribeActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 0) {
                int i = UnsubscribeActivity.this.seconds;
                UnsubscribeActivity unsubscribeActivity = UnsubscribeActivity.this;
                if (i > 0) {
                    UnsubscribeActivity.this.mBtnCode.setText(LanguageUtil.getLanguageByKey(unsubscribeActivity, "hasSent") + " (" + UnsubscribeActivity.this.seconds + "s)");
                    UnsubscribeActivity.access$010(UnsubscribeActivity.this);
                    UnsubscribeActivity.this.mHandler.sendEmptyMessageDelayed(0, 1000L);
                } else {
                    unsubscribeActivity.mBtnCode.setEnabled(true);
                    UnsubscribeActivity.this.mBtnCode.setOnClickListener(UnsubscribeActivity.this);
                    UnsubscribeActivity unsubscribeActivity2 = UnsubscribeActivity.this;
                    LanguageUtil.setText(unsubscribeActivity2, unsubscribeActivity2.mBtnCode, "getVerificationCode");
                }
            }
            return false;
        }
    });

    static /* synthetic */ int access$010(UnsubscribeActivity unsubscribeActivity) {
        int i = unsubscribeActivity.seconds;
        unsubscribeActivity.seconds = i - 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void checkSubmit() {
        String trim = this.mEditEmail.getText().toString().trim();
        String trim2 = this.mEditCode.getText().toString().trim();
        if (TextUtils.isEmpty(trim) || TextUtils.isEmpty(trim2)) {
            this.mBtnSubmit.setEnabled(false);
            this.mBtnSubmit.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_style30_25));
            this.mBtnSubmit.setTextColor(getResources().getColor(R.color.app_style_color50));
        } else {
            this.mBtnSubmit.setEnabled(true);
            this.mBtnSubmit.setBackground(ContextCompat.getDrawable(this, R.drawable.bg_corners_style_25));
            this.mBtnSubmit.setTextColor(getResources().getColor(R.color.white));
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void clearUserChatHistoryDB() {
        try {
            if (MyApplication.getApplication().getmUserModel() != null) {
                this.dbWrite.delete(DatabaseHelper.TABLE_MESSAGE, "sendUserId = ?", new String[]{MyApplication.getApplication().getmUserModel().getUserId() + ""});
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void closeDB() {
        SQLiteDatabase sQLiteDatabase = this.dbWrite;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        DatabaseHelper databaseHelper = this.dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exitAccount() {
        try {
            this.mHandler.removeCallbacksAndMessages(null);
            MyApplication.getApplication().setmUserModel(null);
            String str = (String) SharedPreferencesUtil.get(this, Constans.Language);
            Object obj = SharedPreferencesUtil.get(this, Constans.contactUserId, 0);
            int intValue = obj != null ? ((Integer) obj).intValue() : 0;
            String str2 = (String) SharedPreferencesUtil.get(this, Constans.storeUrl);
            String str3 = (String) SharedPreferencesUtil.get(this, Constans.roomVoice);
            String str4 = (String) SharedPreferencesUtil.get(this, Constans.tempToken);
            int i = intValue;
            String str5 = (String) SharedPreferencesUtil.get(this, Constans.aboutApp);
            String str6 = (String) SharedPreferencesUtil.get(this, Constans.copyright);
            String str7 = (String) SharedPreferencesUtil.get(this, Constans.mainBackgroundUrl);
            String str8 = (String) SharedPreferencesUtil.get(this, Constans.bluetoothConnAddress);
            String str9 = (String) SharedPreferencesUtil.get(this, Constans.ProductAllData);
            SharedPreferencesUtil.clear(this);
            SharedPreferencesUtil.put(this, Constans.Language, str);
            SharedPreferencesUtil.put(this, Constans.contactUserId, Integer.valueOf(i));
            SharedPreferencesUtil.put(this, Constans.storeUrl, str2);
            SharedPreferencesUtil.put(this, Constans.roomVoice, str3);
            SharedPreferencesUtil.put(this, Constans.tempToken, str4);
            SharedPreferencesUtil.put(this, Constans.aboutApp, str5);
            SharedPreferencesUtil.put(this, Constans.copyright, str6);
            SharedPreferencesUtil.put(this, Constans.mainBackgroundUrl, str7);
            SharedPreferencesUtil.put(this, Constans.bluetoothConnAddress, str8);
            SharedPreferencesUtil.put(this, Constans.ProductAllData, str9);
            startActivity(new Intent(this, (Class<?>) LoginActivity.class));
            this.appManager.finishAllOtherActivity(LoginActivity.class);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void getVerCode() {
        this.baseRequest.getApi().emailVerCode("removeAccount", this.mEditEmail.getText().toString().trim()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.UnsubscribeActivity.3
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    UnsubscribeActivity.this.mBtnCode.setEnabled(false);
                    UnsubscribeActivity.this.seconds = 90;
                    UnsubscribeActivity.this.mHandler.sendEmptyMessage(0);
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        DatabaseHelper databaseHelper = new DatabaseHelper(this);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        String str = (String) SharedPreferencesUtil.get(this, "email");
        if (MyApplication.getApplication() != null && MyApplication.getApplication().getmUserModel() != null) {
            str = MyApplication.getApplication().getmUserModel().getEmail();
        }
        if (TextUtils.isEmpty(str)) {
            this.mLayoutEdit.setVisibility(8);
            this.mLayoutEmpty.setVisibility(0);
        } else {
            this.mEditEmail.setText(str);
            this.mLayoutEdit.setVisibility(0);
            this.mLayoutEmpty.setVisibility(8);
        }
        checkSubmit();
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mImgEmail = (ImageView) findViewById(R.id.unsubscribe_img_email);
        this.mImgCode = (ImageView) findViewById(R.id.unsubscribe_img_code);
        this.mEditEmail = (TextView) findViewById(R.id.unsubscribe_edit_email);
        this.mTxtRemoveAccountNote = (TextView) findViewById(R.id.unsubscribe_txt_removeAccountNote);
        this.mEditCode = (EditText) findViewById(R.id.unsubscribe_edit_code);
        this.mBtnCode = (Button) findViewById(R.id.unsubscribe_btn_code);
        this.mBtnSubmit = (Button) findViewById(R.id.unsubscribe_btn_submit);
        this.mLayoutEdit = (LinearLayout) findViewById(R.id.unsubscribe_layout_edit);
        this.mLayoutEmpty = (LinearLayout) findViewById(R.id.unsubscribe_layout_empty);
        this.mImgError = (ImageView) findViewById(R.id.unsubscribe_img_error);
        this.mTxtError = (TextView) findViewById(R.id.unsubscribe_txt_error);
        this.mBtnError = (Button) findViewById(R.id.unsubscribe_btn_error);
        LanguageUtil.setText(this, this.mTitleTxtName, "accountCancellation");
        LanguageUtil.setText(this, this.mBtnError, "bindingEmail");
        LanguageUtil.setHint(this, this.mEditCode, "pleasseEnterVerificationCode");
        LanguageUtil.setText(this, this.mBtnCode, "getVerificationCode");
        LanguageUtil.setText(this, this.mBtnSubmit, "submit");
        SpannableString spannableString = new SpannableString("$  " + LanguageUtil.getLanguageByKey(this, "removeAccountNote"));
        Drawable drawable = ContextCompat.getDrawable(this, R.mipmap.ic_tip);
        drawable.setBounds(0, 0, this.mTxtRemoveAccountNote.getLineHeight(), this.mTxtRemoveAccountNote.getLineHeight());
        spannableString.setSpan(new CenterAlignImageSpan(drawable), 0, 1, 1);
        this.mTxtRemoveAccountNote.setText(spannableString);
        SpannableString spannableString2 = new SpannableString("$  " + LanguageUtil.getLanguageByKey(this, "unboundEmails"));
        Drawable drawable2 = ContextCompat.getDrawable(this, R.mipmap.ic_tip);
        drawable2.setBounds(0, 0, this.mTxtError.getLineHeight(), this.mTxtError.getLineHeight());
        spannableString2.setSpan(new CenterAlignImageSpan(drawable2), 0, 1, 1);
        this.mTxtError.setText(spannableString2);
    }

    private void removeAccount() {
        this.baseRequest.getApi().removeAccount((String) SharedPreferencesUtil.get(this, Constans.Token), this.mEditCode.getText().toString().trim()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.UnsubscribeActivity.4
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    UnsubscribeActivity.this.clearUserChatHistoryDB();
                    UnsubscribeActivity.this.exitAccount();
                }
            }
        });
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mBtnError.setOnClickListener(this);
        this.mBtnCode.setOnClickListener(this);
        this.mBtnSubmit.setOnClickListener(this);
        this.mEditCode.addTextChangedListener(new TextWatcher() { // from class: com.hismith.hismithapp.UnsubscribeActivity.2
            @Override // android.text.TextWatcher
            public void afterTextChanged(Editable editable) {
                UnsubscribeActivity.this.checkSubmit();
            }

            @Override // android.text.TextWatcher
            public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            }

            @Override // android.text.TextWatcher
            public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            }
        });
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        if (i != 222) {
            return;
        }
        String str = (String) SharedPreferencesUtil.get(this, "email");
        if (MyApplication.getApplication() != null && MyApplication.getApplication().getmUserModel() != null) {
            str = MyApplication.getApplication().getmUserModel().getEmail();
        }
        if (TextUtils.isEmpty(str)) {
            this.mLayoutEdit.setVisibility(8);
            this.mLayoutEmpty.setVisibility(0);
        } else {
            this.mEditEmail.setText(str);
            this.mLayoutEdit.setVisibility(0);
            this.mLayoutEmpty.setVisibility(8);
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.unsubscribe_btn_error) {
            startActivityForResult(new Intent(this, (Class<?>) ResetEmailActivity.class), 222);
        } else if (id == R.id.unsubscribe_btn_code) {
            getVerCode();
        } else if (id == R.id.unsubscribe_btn_submit) {
            removeAccount();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_unsubscribe);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        closeDB();
        this.mHandler.removeCallbacksAndMessages(null);
    }
}
