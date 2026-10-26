package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class EditBioActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private EditText bio_edit_content;
    private ProgressBar bio_title_progress;
    private TextView bio_txt_count;
    private TextView bio_txt_post;
    private int maxLength = 120;
    private ImageView title_img_back;

    private void init() {
        this.baseRequest = new RetrofitRequest();
    }

    private void initView() {
        this.title_img_back = (ImageView) findViewById(R.id.title_img_back);
        this.bio_txt_post = (TextView) findViewById(R.id.bio_txt_post);
        this.bio_title_progress = (ProgressBar) findViewById(R.id.bio_title_progress);
        this.bio_edit_content = (EditText) findViewById(R.id.bio_edit_content);
        this.bio_txt_count = (TextView) findViewById(R.id.bio_txt_count);
        LanguageUtil.setHint(this, this.bio_edit_content, "editBio");
        LanguageUtil.setText(this, this.bio_txt_post, "publish");
        if (MyApplication.getApplication().getmUserModel() != null) {
            this.bio_edit_content.setText(MyApplication.getApplication().getmUserModel().getBios());
        }
        this.bio_txt_count.setText(this.bio_edit_content.getText().length() + "/" + this.maxLength);
    }

    private void setListener() {
        this.title_img_back.setOnClickListener(this);
        this.bio_txt_post.setOnClickListener(this);
        this.bio_edit_content.addTextChangedListener(new TextWatcher() { // from class: com.hismith.hismithapp.EditBioActivity.1
            @Override // android.text.TextWatcher
            public void afterTextChanged(Editable editable) {
                EditBioActivity.this.bio_txt_count.setText(EditBioActivity.this.bio_edit_content.getText().length() + "/" + EditBioActivity.this.maxLength);
            }

            @Override // android.text.TextWatcher
            public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            }

            @Override // android.text.TextWatcher
            public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            }
        });
    }

    private void setUserBio() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        final String trim = this.bio_edit_content.getText().toString().trim();
        this.baseRequest.getApi().setUserBIO(str, trim).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.EditBioActivity.2
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                if (EditBioActivity.this.bio_txt_post.getVisibility() == 8) {
                    EditBioActivity.this.bio_title_progress.setVisibility(8);
                    EditBioActivity.this.bio_txt_post.setVisibility(0);
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                if (EditBioActivity.this.bio_txt_post.getVisibility() == 8) {
                    EditBioActivity.this.bio_title_progress.setVisibility(8);
                    EditBioActivity.this.bio_txt_post.setVisibility(0);
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    MyApplication.getApplication().getmUserModel().setBios(trim);
                    EditBioActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    ToastUtil.showShortToastCenter(EditBioActivity.this, LanguageUtil.getLanguageByKey(EditBioActivity.this, "published"));
                    EditBioActivity editBioActivity = EditBioActivity.this;
                    SoftKeyboardUtil.hideSoftKeyboard(editBioActivity, editBioActivity.bio_edit_content);
                    EditBioActivity.this.finish();
                }
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            finish();
        } else if (view.getId() == R.id.bio_txt_post) {
            this.bio_txt_post.setVisibility(8);
            this.bio_title_progress.setVisibility(0);
            setUserBio();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_edit_bio);
        initView();
        init();
        setListener();
    }
}
