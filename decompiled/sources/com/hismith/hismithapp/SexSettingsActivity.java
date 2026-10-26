package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class SexSettingsActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private ImageView setting_img_viewCommunity;
    private ImageView setting_img_viewSex;
    private TextView setting_txt_viewCommunity;
    private TextView setting_txt_viewSex;
    private boolean isViewSex = false;
    private boolean isCommunityHide = false;

    /* JADX INFO: Access modifiers changed from: private */
    public void changeSwitch(ImageView imageView, boolean z) {
        if (z) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_setting_switch_on));
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_setting_switch_off));
        }
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        if (MyApplication.getApplication().getmUserModel() != null && MyApplication.getApplication().getmUserModel().getViewSex() == 1) {
            this.isViewSex = true;
        }
        this.isCommunityHide = ((Boolean) SharedPreferencesUtil.get(this, Constans.COMMUNITY_HIDE, false)).booleanValue();
        changeSwitch(this.setting_img_viewSex, this.isViewSex);
        changeSwitch(this.setting_img_viewCommunity, this.isCommunityHide);
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.setting_txt_viewSex = (TextView) findViewById(R.id.setting_txt_viewSex);
        this.setting_img_viewSex = (ImageView) findViewById(R.id.setting_img_viewSex);
        this.setting_txt_viewCommunity = (TextView) findViewById(R.id.setting_txt_viewCommunity);
        this.setting_img_viewCommunity = (ImageView) findViewById(R.id.setting_img_viewCommunity);
        LanguageUtil.setText(this, this.mTitleTxtName, "sensitiveContent");
        LanguageUtil.setText(this, this.setting_txt_viewSex, "sensitiveReminderTitle");
        LanguageUtil.setText(this, this.setting_txt_viewCommunity, "simpleMode");
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.setting_img_viewSex.setOnClickListener(this);
        this.setting_img_viewCommunity.setOnClickListener(this);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setViewSex() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        boolean z = this.isViewSex;
        Observable<HttpModel> observeOn = this.baseRequest.getApi().setViewSex(str, z ? 1 : 0).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        final int i = z ? 1 : 0;
        observeOn.subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.SexSettingsActivity.1
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 0) {
                    SexSettingsActivity.this.isViewSex = !r3.isViewSex;
                    SexSettingsActivity sexSettingsActivity = SexSettingsActivity.this;
                    sexSettingsActivity.changeSwitch(sexSettingsActivity.setting_img_viewSex, SexSettingsActivity.this.isViewSex);
                    return;
                }
                if (httpModel.getSuccess() != 1 || MyApplication.getApplication().getmUserModel() == null) {
                    return;
                }
                MyApplication.getApplication().getmUserModel().setViewSex(i);
                SharedPreferencesUtil.put(SexSettingsActivity.this, Constans.UserModel, SexSettingsActivity.this.gson.toJson(MyApplication.getApplication().getmUserModel()));
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
        if (id != R.id.setting_img_viewSex) {
            if (id == R.id.setting_img_viewCommunity) {
                boolean z = !this.isCommunityHide;
                this.isCommunityHide = z;
                changeSwitch(this.setting_img_viewCommunity, z);
                SharedPreferencesUtil.put(this, Constans.COMMUNITY_HIDE, Boolean.valueOf(this.isCommunityHide));
                sendBroadcast(new Intent(Constans.RECEIVER_COMMUNITY_HIDE));
                return;
            }
            return;
        }
        boolean z2 = this.isViewSex;
        if (!z2) {
            DialogUtil.showConfirmDialog(this, "prompt", LanguageUtil.getLanguageByKey(this, "sensitiveReminderContent").replace("\\n", "\n"), null, null, new View.OnClickListener() { // from class: com.hismith.hismithapp.SexSettingsActivity.2
                @Override // android.view.View.OnClickListener
                public void onClick(View view2) {
                    SexSettingsActivity.this.isViewSex = !r3.isViewSex;
                    SexSettingsActivity sexSettingsActivity = SexSettingsActivity.this;
                    sexSettingsActivity.changeSwitch(sexSettingsActivity.setting_img_viewSex, SexSettingsActivity.this.isViewSex);
                    SexSettingsActivity.this.setViewSex();
                    DialogUtil.dismissDialog();
                }
            });
            return;
        }
        boolean z3 = !z2;
        this.isViewSex = z3;
        changeSwitch(this.setting_img_viewSex, z3);
        setViewSex();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_single_sex_settings);
        initView();
        init();
        setListener();
    }
}
