package com.hismith.hismithapp.fragment;

import android.content.ContentValues;
import android.content.Intent;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import com.hismith.hismithapp.AboutActivity;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ChatActivity;
import com.hismith.hismithapp.LanguageActivity;
import com.hismith.hismithapp.LoginActivity;
import com.hismith.hismithapp.MessageRemindActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.SecurityActivity;
import com.hismith.hismithapp.SexSettingsActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DataCleanManager;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class SettingFragment extends Fragment implements View.OnClickListener {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbWrite;
    private ImageView mImgAddFriendOn;
    private ImageView mImgPlayLogs;
    private ImageView mImgStealth;
    private LinearLayout mLayoutAboutus;
    private LinearLayout mLayoutChangeLanguage;
    private LinearLayout mLayoutClearCache;
    private LinearLayout mLayoutContactus;
    private LinearLayout mLayoutLogout;
    private LinearLayout mLayoutMessageRemind;
    private LinearLayout mLayoutSecurity;
    private LinearLayout mLayoutViewSex;
    private View mResetLine;
    private TextView mTxtAboutus;
    private TextView mTxtAddFriendOn;
    private TextView mTxtCacheValue;
    private TextView mTxtChangeLanguage;
    private TextView mTxtClearCache;
    private TextView mTxtContactus;
    private TextView mTxtLogout;
    private TextView mTxtMessageRemind;
    private TextView mTxtPlayLogs;
    private TextView mTxtSecurity;
    private TextView mTxtStealth;
    private TextView mTxtViewSex;
    private boolean isAddFriendOn = true;
    private boolean isStealth = false;
    private boolean isPlayLogs = true;

    /* JADX INFO: Access modifiers changed from: private */
    public void changeSwitch(ImageView imageView, boolean z) {
        BaseActivity baseActivity = this.baseActivity;
        if (z) {
            imageView.setImageDrawable(ContextCompat.getDrawable(baseActivity, R.mipmap.ic_setting_switch_on));
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(baseActivity, R.mipmap.ic_setting_switch_off));
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

    private void getHismithInfo() {
        if (MyApplication.getApplication().getmHismithModel() != null) {
            Intent intent = new Intent(this.baseActivity, (Class<?>) ChatActivity.class);
            intent.putExtra(Constans.UserModel, MyApplication.getApplication().getmHismithModel());
            intent.putExtra("firstChat", 0);
            startActivity(intent);
            return;
        }
        String str = (String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token);
        Object obj = SharedPreferencesUtil.get(this.baseActivity, Constans.contactUserId, 1);
        Observable<UserModel> observeOn = this.baseRequest.getApi().getUserInfo(str, Integer.valueOf(obj != null ? ((Integer) obj).intValue() : 1)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<UserModel>(baseActivity, baseActivity.progressDialog) { // from class: com.hismith.hismithapp.fragment.SettingFragment.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel != null) {
                    SettingFragment.this.insertDBMessageUser(userModel);
                    MyApplication.getApplication().setmHismithModel(userModel);
                    Intent intent2 = new Intent(SettingFragment.this.baseActivity, (Class<?>) ChatActivity.class);
                    intent2.putExtra(Constans.UserModel, userModel);
                    intent2.putExtra("firstChat", 0);
                    SettingFragment.this.startActivity(intent2);
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        DatabaseHelper databaseHelper = new DatabaseHelper(this.baseActivity);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        if (MyApplication.getApplication().getmUserModel() != null) {
            if (MyApplication.getApplication().getmUserModel().getChatUpState() == 0) {
                this.isAddFriendOn = false;
            }
            if (MyApplication.getApplication().getmUserModel().getStealth() == 1) {
                this.isStealth = true;
            }
            if (MyApplication.getApplication().getmUserModel().getPlayLogs() == 0) {
                this.isPlayLogs = false;
            }
        }
        changeSwitch(this.mImgAddFriendOn, this.isAddFriendOn);
        changeSwitch(this.mImgStealth, this.isStealth);
        changeSwitch(this.mImgPlayLogs, this.isPlayLogs);
        this.mTxtCacheValue.setText(DataCleanManager.getCacheSize(this.baseActivity));
    }

    private void initView(View view) {
        this.mTxtMessageRemind = (TextView) view.findViewById(R.id.setting_txt_messageRemind);
        this.mTxtAddFriendOn = (TextView) view.findViewById(R.id.setting_txt_addFriendOn);
        this.mTxtStealth = (TextView) view.findViewById(R.id.setting_txt_stealth);
        this.mTxtPlayLogs = (TextView) view.findViewById(R.id.setting_txt_playLogs);
        this.mTxtViewSex = (TextView) view.findViewById(R.id.setting_txt_viewSex);
        this.mTxtClearCache = (TextView) view.findViewById(R.id.setting_txt_clearCache);
        this.mTxtCacheValue = (TextView) view.findViewById(R.id.setting_txt_cacheValue);
        this.mTxtChangeLanguage = (TextView) view.findViewById(R.id.setting_txt_changeLanguage);
        this.mTxtSecurity = (TextView) view.findViewById(R.id.setting_txt_security);
        this.mTxtAboutus = (TextView) view.findViewById(R.id.setting_txt_aboutus);
        this.mTxtContactus = (TextView) view.findViewById(R.id.setting_txt_contactus);
        this.mTxtLogout = (TextView) view.findViewById(R.id.setting_txt_logout);
        this.mImgAddFriendOn = (ImageView) view.findViewById(R.id.setting_img_addFriendOn);
        this.mImgStealth = (ImageView) view.findViewById(R.id.setting_img_stealth);
        this.mImgPlayLogs = (ImageView) view.findViewById(R.id.setting_img_playLogs);
        this.mLayoutViewSex = (LinearLayout) view.findViewById(R.id.setting_layout_viewSex);
        this.mLayoutClearCache = (LinearLayout) view.findViewById(R.id.setting_layout_clearCache);
        this.mLayoutMessageRemind = (LinearLayout) view.findViewById(R.id.setting_layout_messageRemind);
        this.mLayoutChangeLanguage = (LinearLayout) view.findViewById(R.id.setting_layout_changeLanguage);
        this.mLayoutSecurity = (LinearLayout) view.findViewById(R.id.setting_layout_security);
        this.mLayoutAboutus = (LinearLayout) view.findViewById(R.id.setting_layout_aboutus);
        this.mLayoutContactus = (LinearLayout) view.findViewById(R.id.setting_layout_contactus);
        this.mLayoutLogout = (LinearLayout) view.findViewById(R.id.setting_layout_logout);
        this.mResetLine = view.findViewById(R.id.setting_reset_line);
        LanguageUtil.setText(this.baseActivity, this.mTxtMessageRemind, "messageRemind");
        LanguageUtil.setText(this.baseActivity, this.mTxtAddFriendOn, "addFriendOn");
        LanguageUtil.setText(this.baseActivity, this.mTxtStealth, "hideOnline");
        LanguageUtil.setText(this.baseActivity, this.mTxtPlayLogs, "playRecord");
        LanguageUtil.setText(this.baseActivity, this.mTxtViewSex, "sensitiveContent");
        LanguageUtil.setText(this.baseActivity, this.mTxtClearCache, "clearCache");
        LanguageUtil.setText(this.baseActivity, this.mTxtChangeLanguage, "changeLanguage");
        LanguageUtil.setText(this.baseActivity, this.mTxtSecurity, "accountSecurity");
        LanguageUtil.setText(this.baseActivity, this.mTxtAboutus, "aboutus");
        LanguageUtil.setText(this.baseActivity, this.mTxtContactus, "contactUs");
        LanguageUtil.setText(this.baseActivity, this.mTxtLogout, "logout");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageUser(UserModel userModel) {
        if (userModel == null) {
            return;
        }
        try {
            ContentValues contentValues = new ContentValues();
            contentValues.put("userId", Integer.valueOf(userModel.getUserId()));
            contentValues.put("king", Integer.valueOf(userModel.getKing()));
            contentValues.put("gradeIcon", userModel.getGradeIcon());
            contentValues.put("name", userModel.getName());
            contentValues.put("gender", Integer.valueOf(userModel.getGender()));
            contentValues.put("headIcon", userModel.getHeadIcon());
            this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void setChatUpState() {
        String str = (String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token);
        final Integer valueOf = Integer.valueOf(this.isAddFriendOn ? 1 : 0);
        Observable<HttpModel> observeOn = this.baseRequest.getApi().setChatUpState(str, valueOf).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.fragment.SettingFragment.1
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 0) {
                    SettingFragment.this.isAddFriendOn = !r3.isAddFriendOn;
                    SettingFragment settingFragment = SettingFragment.this;
                    settingFragment.changeSwitch(settingFragment.mImgAddFriendOn, SettingFragment.this.isAddFriendOn);
                    return;
                }
                if (httpModel.getSuccess() != 1 || MyApplication.getApplication().getmUserModel() == null) {
                    return;
                }
                MyApplication.getApplication().getmUserModel().setChatUpState(valueOf.intValue());
                SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.UserModel, SettingFragment.this.baseActivity.gson.toJson(MyApplication.getApplication().getmUserModel()));
            }
        });
    }

    private void setListener() {
        this.mLayoutMessageRemind.setOnClickListener(this);
        this.mLayoutAboutus.setOnClickListener(this);
        this.mLayoutChangeLanguage.setOnClickListener(this);
        this.mLayoutSecurity.setOnClickListener(this);
        this.mLayoutContactus.setOnClickListener(this);
        this.mImgAddFriendOn.setOnClickListener(this);
        this.mImgStealth.setOnClickListener(this);
        this.mImgPlayLogs.setOnClickListener(this);
        this.mLayoutViewSex.setOnClickListener(this);
        this.mLayoutClearCache.setOnClickListener(this);
        this.mLayoutLogout.setOnClickListener(this);
    }

    private void setPlayLogs() {
        String str = (String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token);
        final Integer valueOf = Integer.valueOf(this.isPlayLogs ? 1 : 0);
        Observable<HttpModel> observeOn = this.baseRequest.getApi().setPlayLogs(str, valueOf).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.fragment.SettingFragment.3
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 0) {
                    SettingFragment.this.isPlayLogs = !r3.isPlayLogs;
                    SettingFragment settingFragment = SettingFragment.this;
                    settingFragment.changeSwitch(settingFragment.mImgPlayLogs, SettingFragment.this.isPlayLogs);
                    return;
                }
                if (httpModel.getSuccess() != 1 || MyApplication.getApplication().getmUserModel() == null) {
                    return;
                }
                MyApplication.getApplication().getmUserModel().setPlayLogs(valueOf.intValue());
                SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.UserModel, SettingFragment.this.baseActivity.gson.toJson(MyApplication.getApplication().getmUserModel()));
            }
        });
    }

    private void setStealth() {
        String str = (String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token);
        final Integer valueOf = Integer.valueOf(this.isStealth ? 1 : 0);
        Observable<HttpModel> observeOn = this.baseRequest.getApi().setStealth(str, valueOf).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.fragment.SettingFragment.2
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 0) {
                    SettingFragment.this.isStealth = !r3.isStealth;
                    SettingFragment settingFragment = SettingFragment.this;
                    settingFragment.changeSwitch(settingFragment.mImgStealth, SettingFragment.this.isStealth);
                    return;
                }
                if (httpModel.getSuccess() != 1 || MyApplication.getApplication().getmUserModel() == null) {
                    return;
                }
                MyApplication.getApplication().getmUserModel().setStealth(valueOf.intValue());
                SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.UserModel, SettingFragment.this.baseActivity.gson.toJson(MyApplication.getApplication().getmUserModel()));
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.setting_layout_messageRemind) {
            startActivity(new Intent(this.baseActivity, (Class<?>) MessageRemindActivity.class));
            return;
        }
        if (id == R.id.setting_img_addFriendOn) {
            boolean z = !this.isAddFriendOn;
            this.isAddFriendOn = z;
            changeSwitch(this.mImgAddFriendOn, z);
            setChatUpState();
            return;
        }
        if (id == R.id.setting_img_stealth) {
            boolean z2 = !this.isStealth;
            this.isStealth = z2;
            changeSwitch(this.mImgStealth, z2);
            setStealth();
            return;
        }
        if (id == R.id.setting_img_playLogs) {
            boolean z3 = !this.isPlayLogs;
            this.isPlayLogs = z3;
            changeSwitch(this.mImgPlayLogs, z3);
            setPlayLogs();
            return;
        }
        if (id == R.id.setting_layout_viewSex) {
            startActivity(new Intent(this.baseActivity, (Class<?>) SexSettingsActivity.class));
            return;
        }
        if (id == R.id.setting_layout_clearCache) {
            DataCleanManager.cleanApplicationData(this.baseActivity, new String[0]);
            this.mTxtCacheValue.setText(DataCleanManager.getCacheSize(this.baseActivity));
            return;
        }
        if (id == R.id.setting_layout_changeLanguage) {
            startActivity(new Intent(this.baseActivity, (Class<?>) LanguageActivity.class));
            return;
        }
        if (id == R.id.setting_layout_security) {
            startActivity(new Intent(this.baseActivity, (Class<?>) SecurityActivity.class));
            return;
        }
        if (id == R.id.setting_layout_aboutus) {
            startActivity(new Intent(this.baseActivity, (Class<?>) AboutActivity.class));
        } else if (id == R.id.setting_layout_contactus) {
            getHismithInfo();
        } else if (id == R.id.setting_layout_logout) {
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "areYouSureSignouAccount", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.SettingFragment.5
                @Override // android.view.View.OnClickListener
                public void onClick(View view2) {
                    try {
                        MyApplication.getApplication().setmUserModel(null);
                        String str = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.Language);
                        Object obj = SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.contactUserId, 0);
                        int intValue = obj != null ? ((Integer) obj).intValue() : 0;
                        String str2 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.storeUrl);
                        String str3 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.roomVoice);
                        String str4 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.tempToken);
                        int i = intValue;
                        String str5 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.aboutApp);
                        String str6 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.copyright);
                        String str7 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.mainBackgroundUrl);
                        String str8 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.bluetoothConnAddress);
                        String str9 = (String) SharedPreferencesUtil.get(SettingFragment.this.baseActivity, Constans.ProductAllData);
                        SharedPreferencesUtil.clear(SettingFragment.this.baseActivity);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.Language, str);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.contactUserId, Integer.valueOf(i));
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.storeUrl, str2);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.roomVoice, str3);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.tempToken, str4);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.aboutApp, str5);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.copyright, str6);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.mainBackgroundUrl, str7);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.bluetoothConnAddress, str8);
                        SharedPreferencesUtil.put(SettingFragment.this.baseActivity, Constans.ProductAllData, str9);
                        BleManager.getInstance(SettingFragment.this.baseActivity).stopLeScan();
                        BleManager.getInstance(SettingFragment.this.baseActivity).release();
                        SettingFragment.this.baseActivity.startActivity(new Intent(SettingFragment.this.baseActivity, (Class<?>) LoginActivity.class));
                        SettingFragment.this.baseActivity.appManager.finishAllOtherActivity(LoginActivity.class);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_setting, (ViewGroup) null);
        initView(inflate);
        init();
        setListener();
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        closeDB();
        DialogUtil.dismissDialog();
    }
}
