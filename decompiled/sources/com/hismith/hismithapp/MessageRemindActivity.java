package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class MessageRemindActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private ImageView mImgFollowMessage;
    private ImageView mImgInteractiveMessage;
    private ImageView mImgMyMessage;
    private ImageView mImgProductAds;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtFollowMessage;
    private TextView mTxtInteractiveMessage;
    private TextView mTxtMyMessage;
    private TextView mTxtProductAds;
    private boolean isInteractiveMessage = true;
    private boolean isFollowMessage = true;
    private boolean isMyMessage = true;
    private boolean isProductAds = true;

    private void changeSwitch(ImageView imageView, boolean z) {
        if (z) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_setting_switch_on));
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_setting_switch_off));
        }
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        if (MyApplication.getApplication().getmUserModel() != null) {
            if (MyApplication.getApplication().getmUserModel().getMutuallyRemind() == 0) {
                this.isInteractiveMessage = false;
            }
            if (MyApplication.getApplication().getmUserModel().getFocusRemind() == 0) {
                this.isFollowMessage = false;
            }
            if (MyApplication.getApplication().getmUserModel().getMessageRemind() == 0) {
                this.isMyMessage = false;
            }
            if (MyApplication.getApplication().getmUserModel().getProductAds() == 0) {
                this.isProductAds = false;
            }
        }
        changeSwitch(this.mImgInteractiveMessage, this.isInteractiveMessage);
        changeSwitch(this.mImgFollowMessage, this.isFollowMessage);
        changeSwitch(this.mImgMyMessage, this.isMyMessage);
        changeSwitch(this.mImgProductAds, this.isProductAds);
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTxtInteractiveMessage = (TextView) findViewById(R.id.setting_txt_interactiveMessage);
        this.mTxtFollowMessage = (TextView) findViewById(R.id.setting_txt_followMessage);
        this.mTxtMyMessage = (TextView) findViewById(R.id.setting_txt_myMessage);
        this.mImgInteractiveMessage = (ImageView) findViewById(R.id.setting_img_interactiveMessage);
        this.mImgFollowMessage = (ImageView) findViewById(R.id.setting_img_followMessage);
        this.mImgMyMessage = (ImageView) findViewById(R.id.setting_img_myMessage);
        this.mTxtProductAds = (TextView) findViewById(R.id.setting_txt_productAds);
        this.mImgProductAds = (ImageView) findViewById(R.id.setting_img_productAds);
        LanguageUtil.setText(this, this.mTitleTxtName, "messageSetting");
        LanguageUtil.setText(this, this.mTxtInteractiveMessage, "interactiveMessage");
        LanguageUtil.setText(this, this.mTxtFollowMessage, "followMessage");
        LanguageUtil.setText(this, this.mTxtMyMessage, "myMessage");
        LanguageUtil.setText(this, this.mTxtProductAds, "productAds");
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mImgInteractiveMessage.setOnClickListener(this);
        this.mImgFollowMessage.setOnClickListener(this);
        this.mImgMyMessage.setOnClickListener(this);
        this.mImgProductAds.setOnClickListener(this);
    }

    private void setRemind() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        Integer valueOf = Integer.valueOf(this.isInteractiveMessage ? 1 : 0);
        Integer valueOf2 = Integer.valueOf(this.isFollowMessage ? 1 : 0);
        Integer valueOf3 = Integer.valueOf(this.isMyMessage ? 1 : 0);
        boolean z = this.isProductAds;
        Observable<HttpModel> observeOn = this.baseRequest.getApi().setRemind(str, valueOf, valueOf2, valueOf3, Integer.valueOf(z ? 1 : 0)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        final int i = z ? 1 : 0;
        observeOn.subscribe(new RxSubscribe<HttpModel>(MyApplication.getApplication(), null, false) { // from class: com.hismith.hismithapp.MessageRemindActivity.2
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() != 1) {
                    httpModel.getSuccess();
                } else if (i == 0) {
                    SharedPreferencesUtil.remove(MessageRemindActivity.this, Constans.adsVersion);
                    CacheUtil.deleteAds(MessageRemindActivity.this);
                }
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.setting_img_interactiveMessage) {
            boolean z = !this.isInteractiveMessage;
            this.isInteractiveMessage = z;
            changeSwitch(this.mImgInteractiveMessage, z);
            if (MyApplication.getApplication().getmUserModel() != null) {
                MyApplication.getApplication().getmUserModel().setMutuallyRemind(this.isInteractiveMessage ? 1 : 0);
            }
            setRemind();
            return;
        }
        if (id == R.id.setting_img_followMessage) {
            boolean z2 = !this.isFollowMessage;
            this.isFollowMessage = z2;
            changeSwitch(this.mImgFollowMessage, z2);
            if (MyApplication.getApplication().getmUserModel() != null) {
                MyApplication.getApplication().getmUserModel().setFocusRemind(this.isFollowMessage ? 1 : 0);
            }
            setRemind();
            return;
        }
        if (id == R.id.setting_img_myMessage) {
            boolean z3 = !this.isMyMessage;
            this.isMyMessage = z3;
            changeSwitch(this.mImgMyMessage, z3);
            if (MyApplication.getApplication().getmUserModel() != null) {
                MyApplication.getApplication().getmUserModel().setMessageRemind(this.isMyMessage ? 1 : 0);
            }
            setRemind();
            return;
        }
        if (id == R.id.setting_img_productAds) {
            boolean z4 = !this.isProductAds;
            this.isProductAds = z4;
            changeSwitch(this.mImgProductAds, z4);
            if (MyApplication.getApplication().getmUserModel() != null) {
                MyApplication.getApplication().getmUserModel().setProductAds(this.isProductAds ? 1 : 0);
            }
            setRemind();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.MessageRemindActivity.1
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                MessageRemindActivity.this.finish();
            }
        });
        setContentView(R.layout.activity_message_remind);
        initView();
        init();
        setListener();
    }
}
