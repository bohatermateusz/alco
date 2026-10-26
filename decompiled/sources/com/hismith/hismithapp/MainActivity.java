package com.hismith.hismithapp;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.bluetooth.BluetoothGatt;
import android.content.BroadcastReceiver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import android.view.animation.LinearInterpolator;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.ListFragment;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.viewpager2.widget.ViewPager2;
import com.facebook.share.internal.ShareConstants;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.MainActivity;
import com.hismith.hismithapp.adapter.FragmentAdapter2;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DBMessageLastUtil;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.fragment.CommunityFragment2;
import com.hismith.hismithapp.fragment.PersonalFragment;
import com.hismith.hismithapp.fragment.PlayFragment;
import com.hismith.hismithapp.fragment.SettingFragment;
import com.hismith.hismithapp.fragment.ShopFragment;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RemoteFloatingManager;
import com.hismith.hismithapp.manager.UpgradeManager;
import com.hismith.hismithapp.model.ComplaintReasonsModel;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.TopicCategoriesModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.observer.AppBackgroundUtil;
import com.hismith.hismithapp.service.HuDongPlayService;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.BlePermissionUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import com.yhao.floatwindow.FloatWindow;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import io.reactivex.observers.DisposableObserver;
import io.reactivex.schedulers.Schedulers;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.TimeZone;
import okhttp3.ResponseBody;

/* loaded from: classes3.dex */
public class MainActivity extends BaseActivity implements View.OnClickListener, BleManager.OnConnectStateListener {
    public static final int WHAT_MESSAGE_NEW = 2;
    public static final int WHAT_USER_INFO = 1;
    public RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbRead;
    private SQLiteDatabase dbWrite;
    private BluetoothGatt mBluetoothGatt;
    private CommunityFragment2 mCommunityFragment;
    private FragmentAdapter2 mFragmentAdapter;
    private ValueAnimator mHideValueAnimator;
    private ImageView mImgCommunity;
    private ImageView mImgMy;
    private ImageView mImgMyDot;
    private ImageView mImgPlay;
    private ImageView mImgStore;
    private LinearLayout mLayoutCommunity;
    private LinearLayout mLayoutMenu;
    private LinearLayout mLayoutMy;
    private LinearLayout mLayoutPlay;
    private LinearLayout mLayoutRoot;
    private LinearLayout mLayoutStore;
    private PersonalFragment mPersonalFragment;
    private PlayFragment mPlayFragment;
    private SettingFragment mSettingFragment;
    private ShopFragment mShopFragment;
    private ValueAnimator mShowValueAnimator;
    private TextView mTxtCommunity;
    private TextView mTxtMy;
    private TextView mTxtPlay;
    private TextView mTxtStore;
    private ViewPager2 mViewPager;
    private List<Fragment> fragments = new ArrayList();
    private boolean scrollable = true;
    private int lastInteractive = 0;
    private boolean isShowStore = false;
    private boolean isDeviceExpired = true;
    private boolean isBackground = false;
    public int messageDelayed = 5000;
    private long lastMessageTime = 0;
    private long mPressedTime = 0;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    public Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.MainActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                int i2 = message.arg1;
                return false;
            }
            if (i != 2 || MyApplication.getApplication().getmUserModel() == null) {
                return false;
            }
            MainActivity.this.messageNew(MainActivity.this.getMessageLastId());
            return false;
        }
    });
    private int menuHeight = 0;
    private boolean isHide = false;
    private BroadcastReceiver mReceiver = new AnonymousClass16();

    /* renamed from: com.hismith.hismithapp.MainActivity$16, reason: invalid class name */
    class AnonymousClass16 extends BroadcastReceiver {
        AnonymousClass16() {
        }

        /* renamed from: lambda$onReceive$0$com-hismith-hismithapp-MainActivity$16, reason: not valid java name */
        /* synthetic */ void m564lambda$onReceive$0$comhismithhismithappMainActivity$16() {
            MainActivity.this.initTab();
        }

        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals(Constans.RECEIVER_USER_INFO_CHANGE)) {
                if (MainActivity.this.mPersonalFragment != null) {
                    MainActivity.this.mPersonalFragment.initUserInfo(MyApplication.getApplication().getmUserModel());
                }
                MainActivity.this.getMyInfo();
                return;
            }
            if (Constans.RECEIVER_BLE_CONN_STATE.equals(action)) {
                final int intExtra = intent.getIntExtra("ble_state", 0);
                MainActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.MainActivity.16.1
                    @Override // java.lang.Runnable
                    public void run() {
                        int i = intExtra;
                        AnonymousClass16 anonymousClass16 = AnonymousClass16.this;
                        if (i != 2) {
                            if (MainActivity.this.mPlayFragment == null || !MainActivity.this.mPlayFragment.isAdded()) {
                                return;
                            }
                            MainActivity.this.mPlayFragment.changeBluetooth(0);
                            return;
                        }
                        MainActivity.this.initTab();
                        if (MainActivity.this.mPlayFragment == null || !MainActivity.this.mPlayFragment.isAdded()) {
                            return;
                        }
                        MainActivity.this.mPlayFragment.changeBluetooth(1);
                    }
                });
                return;
            }
            if (Constans.RECEIVER_BLE_CODE_STATE.equals(action)) {
                MainActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.MainActivity.16.2
                    @Override // java.lang.Runnable
                    public void run() {
                        if (MainActivity.this.mPlayFragment == null || !MainActivity.this.mPlayFragment.isAdded()) {
                            return;
                        }
                        MainActivity.this.mPlayFragment.autoRefreshTop();
                    }
                });
                return;
            }
            if (!Constans.RECEIVER_MESSAGE_READ.equals(action)) {
                if (Constans.RECEIVER_COMMUNITY_HIDE.equals(action)) {
                    MainActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.MainActivity$16$$ExternalSyntheticLambda0
                        @Override // java.lang.Runnable
                        public final void run() {
                            MainActivity.AnonymousClass16.this.m564lambda$onReceive$0$comhismithhismithappMainActivity$16();
                        }
                    });
                }
            } else {
                if (intent.getBooleanExtra("isUnRead", false) || MainActivity.this.lastInteractive != 0) {
                    return;
                }
                MainActivity.this.mImgMyDot.setVisibility(8);
            }
        }
    }

    private void closeDB() {
        LogUtil.e("注销过程 MainActivity closeDB()");
        SQLiteDatabase sQLiteDatabase = this.dbRead;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
            LogUtil.e("注销过程 dbRead closeDB()");
        }
        SQLiteDatabase sQLiteDatabase2 = this.dbWrite;
        if (sQLiteDatabase2 != null) {
            sQLiteDatabase2.close();
            LogUtil.e("注销过程 dbWrite closeDB()");
        }
        DatabaseHelper databaseHelper = this.dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
            LogUtil.e("注销过程 dbHepler closeDB()");
        }
    }

    private void colseSocket() {
        if (SocketManager.getInstance().getWebSocket() != null) {
            LogUtil.e("WebSocket 发送：关闭");
            SocketManager.getInstance().setOnSocketNotifyListener(null);
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void downloadAds(final String str, String str2, final String str3) {
        LogUtil.e("开始下载广告页 版本号：" + str + ", 图片URL:" + str2 + ", 网页标签：" + str3);
        this.baseRequest.getApi().download(str2).subscribeOn(Schedulers.io()).observeOn(Schedulers.newThread()).subscribe(new DisposableObserver<ResponseBody>() { // from class: com.hismith.hismithapp.MainActivity.10
            @Override // io.reactivex.Observer
            public void onComplete() {
                LogUtil.e("Ads下载响应完成");
            }

            @Override // io.reactivex.Observer
            public void onError(Throwable th) {
                LogUtil.e("Ads下载响应错误:" + th.getMessage());
            }

            @Override // io.reactivex.Observer
            public void onNext(ResponseBody responseBody) {
                LogUtil.e("Ads下载响应成功");
                try {
                    byte[] bytes = responseBody.bytes();
                    LogUtil.e("Ads图片数据：" + bytes.length);
                    CacheUtil.saveAds(MainActivity.this, str, BitmapFactory.decodeByteArray(bytes, 0, bytes.length), str3);
                } catch (Exception e) {
                    LogUtil.e("Ads下载 出错:" + e.getMessage());
                    e.printStackTrace();
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public int getMessageLastId() {
        try {
            int messageLastId = DBMessageLastUtil.getMessageLastId();
            return messageLastId == 0 ? DBMessageUtil.getMessageLastId() : messageLastId;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getMyInfo() {
        this.baseRequest.getApi().getMyInfo((String) SharedPreferencesUtil.get(this, Constans.Token), MyApplication.FCM_TOKEN, AppUtil.getAppVersionName(this)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.MainActivity.9
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    if (MainActivity.this.mPersonalFragment != null) {
                        MainActivity.this.mPersonalFragment.initUserInfo(userModel);
                    }
                    MyApplication.getApplication().setmUserModel(userModel);
                    if (!MainActivity.this.mHandler.hasMessages(2)) {
                        MainActivity.this.mHandler.sendEmptyMessage(2);
                    }
                    SharedPreferencesUtil.put(MainActivity.this, Constans.adsVersion, userModel.getAdsVersion());
                    if (TextUtils.isEmpty(userModel.getAdsVersion())) {
                        LogUtil.e("Ads没有返回广告页");
                        CacheUtil.deleteAds(MainActivity.this);
                    } else if (CacheUtil.adsVersionExists(MainActivity.this, userModel.getAdsVersion())) {
                        LogUtil.e("Ads已存在版本号" + userModel.getAdsVersion() + "的广告页");
                    } else {
                        MainActivity.this.downloadAds(userModel.getAdsVersion(), userModel.getAdsImage(), userModel.getAdsContent());
                    }
                }
            }
        });
    }

    private void getUserInfoToChat(int i) {
        this.baseRequest.getApi().getUserInfo((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.MainActivity.11
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    Intent intent = new Intent(MainActivity.this, (Class<?>) ChatActivity.class);
                    intent.putExtra(Constans.UserModel, userModel);
                    intent.putExtra("firstChat", DBMessageUtil.getMessageCountByUser(userModel.getUserId()) > 0 ? 0 : 1);
                    MainActivity.this.startActivity(intent);
                }
            }
        });
    }

    /* JADX WARN: Code restructure failed: missing block: B:21:0x00b0, code lost:
    
        if ("".equals(r4) != false) goto L30;
     */
    /* JADX WARN: Code restructure failed: missing block: B:22:0x00b2, code lost:
    
        getUserInfoToChat(java.lang.Integer.valueOf(r4).intValue());
     */
    /* JADX WARN: Code restructure failed: missing block: B:23:0x00bd, code lost:
    
        return;
     */
    /* JADX WARN: Code restructure failed: missing block: B:24:?, code lost:
    
        return;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void goViewToPush() {
        try {
            if (getIntent() == null || getIntent().getExtras() == null) {
                LogUtil.e("getIntent():null");
                return;
            }
            String string = getIntent().getExtras().getString("category");
            String string2 = getIntent().getExtras().getString("sourceId");
            String string3 = getIntent().getExtras().getString("resultId");
            String string4 = getIntent().getExtras().getString("userId");
            LogUtil.e("category:" + string + ", sourceId:" + string2 + ", resultId:" + string3 + ", userId:" + string4);
            if (!ShareConstants.WEB_DIALOG_PARAM_MESSAGE.equals(string) && !"firstChat".equals(string)) {
                if (!"addfocus".equals(string)) {
                    "interactive".equals(string);
                    return;
                }
                Intent intent = new Intent(this, (Class<?>) FansFocusActivity.class);
                intent.putExtra("fansMode", 1);
                startActivity(intent);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initBluetoothPermission() {
        LogUtil.e("BlePermissionUtil.isHarmony(this):" + BlePermissionUtil.isHarmony(this));
        Disposable subscribe = new RxPermissions(this).requestEachCombined((Build.VERSION.SDK_INT < 31 || BlePermissionUtil.isHarmony(this)) ? new String[]{"android.permission.ACCESS_COARSE_LOCATION", "android.permission.ACCESS_FINE_LOCATION"} : new String[]{"android.permission.BLUETOOTH_SCAN", "android.permission.BLUETOOTH_CONNECT"}).subscribe(new Consumer() { // from class: com.hismith.hismithapp.MainActivity$$ExternalSyntheticLambda1
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                MainActivity.this.m563xac8a8fd8((Permission) obj);
            }
        });
        if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initTab() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.storeUrl, "");
        this.isDeviceExpired = AppUtil.isDeviceExpired(this);
        if (TextUtils.isEmpty(str) || this.isDeviceExpired) {
            this.isShowStore = false;
        } else {
            this.isShowStore = true;
        }
        boolean z = this.isDeviceExpired;
        LinearLayout linearLayout = this.mLayoutCommunity;
        if (z) {
            linearLayout.setVisibility(8);
        } else {
            linearLayout.setVisibility(0);
        }
        boolean z2 = this.isShowStore;
        LinearLayout linearLayout2 = this.mLayoutStore;
        if (z2) {
            linearLayout2.setVisibility(0);
        } else {
            linearLayout2.setVisibility(8);
        }
        ArrayList arrayList = new ArrayList();
        arrayList.add(LanguageUtil.getLanguageByKey(this, "play"));
        if (!this.isDeviceExpired) {
            arrayList.add(LanguageUtil.getLanguageByKey(this, "community"));
        }
        if (this.isShowStore) {
            arrayList.add(LanguageUtil.getLanguageByKey(this, "store"));
        }
        arrayList.add(LanguageUtil.getLanguageByKey(this, "me"));
        ArrayList arrayList2 = new ArrayList();
        this.fragments = arrayList2;
        arrayList2.add(this.mPlayFragment);
        if (!this.isDeviceExpired) {
            this.fragments.add(this.mCommunityFragment);
        }
        if (this.isShowStore) {
            this.fragments.add(this.mShopFragment);
        }
        this.fragments.add(this.mPersonalFragment);
        for (int size = this.fragments.size(); size < arrayList.size(); size++) {
            this.fragments.add(new ListFragment());
        }
        FragmentAdapter2 fragmentAdapter2 = this.mFragmentAdapter;
        if (fragmentAdapter2 != null) {
            fragmentAdapter2.updateData(this.fragments, arrayList);
        }
        this.mViewPager.setOffscreenPageLimit(this.fragments.size());
        this.mViewPager.setCurrentItem(0);
    }

    private void initValue(Bundle bundle) {
        UserModel userModel;
        this.baseRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_BASE);
        DatabaseHelper databaseHelper = new DatabaseHelper(this);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        this.dbRead = this.dbHepler.getReadableDatabase();
        this.sdf.setTimeZone(TimeZone.getTimeZone("GMT+00:00"));
        try {
            userModel = (UserModel) this.gson.fromJson((String) SharedPreferencesUtil.get(this, Constans.UserModel), UserModel.class);
        } catch (Exception e) {
            e.printStackTrace();
            userModel = null;
        }
        if (userModel != null) {
            MyApplication.getApplication().setmUserModel(userModel);
            if (this.mHandler.hasMessages(2)) {
                this.mHandler.removeMessages(2);
            }
            this.mHandler.sendEmptyMessage(2);
        }
        if (bundle != null) {
            List<Fragment> fragments = getSupportFragmentManager().getFragments();
            for (int i = 0; i < fragments.size(); i++) {
                Fragment fragment = fragments.get(i);
                if (fragment instanceof PersonalFragment) {
                    this.mPersonalFragment = (PersonalFragment) fragment;
                } else if (fragment instanceof PlayFragment) {
                    this.mPlayFragment = (PlayFragment) fragment;
                } else if (fragment instanceof ShopFragment) {
                    this.mShopFragment = (ShopFragment) fragment;
                } else if (fragment instanceof CommunityFragment2) {
                    this.mCommunityFragment = (CommunityFragment2) fragment;
                } else if (fragment instanceof SettingFragment) {
                    this.mSettingFragment = (SettingFragment) fragment;
                }
            }
        } else {
            this.mPersonalFragment = new PersonalFragment();
            this.mPlayFragment = new PlayFragment();
            this.mShopFragment = new ShopFragment();
            this.mCommunityFragment = new CommunityFragment2();
            this.mSettingFragment = new SettingFragment();
        }
        this.menuHeight = (int) getResources().getDimension(R.dimen.main_layout_menu_height);
        initViewPager();
        initTab();
        selectTab(0);
        initBluetoothPermission();
        getMyInfo();
        topicCategories();
        complaintReasons();
        makeFilter();
        goViewToPush();
        requestNotificationPermission();
    }

    private void initView() {
        this.mLayoutRoot = (LinearLayout) findViewById(R.id.layout_root);
        this.mViewPager = (ViewPager2) findViewById(R.id.main_viewpager);
        this.mLayoutMenu = (LinearLayout) findViewById(R.id.main_layout_menu);
        this.mLayoutPlay = (LinearLayout) findViewById(R.id.main_layout_play);
        this.mLayoutStore = (LinearLayout) findViewById(R.id.main_layout_store);
        this.mLayoutCommunity = (LinearLayout) findViewById(R.id.main_layout_community);
        this.mLayoutMy = (LinearLayout) findViewById(R.id.main_layout_my);
        this.mImgPlay = (ImageView) findViewById(R.id.main_img_play);
        this.mImgStore = (ImageView) findViewById(R.id.main_img_store);
        this.mImgCommunity = (ImageView) findViewById(R.id.main_img_community);
        this.mImgMy = (ImageView) findViewById(R.id.main_img_my);
        this.mTxtPlay = (TextView) findViewById(R.id.main_txt_play);
        this.mTxtStore = (TextView) findViewById(R.id.main_txt_store);
        this.mTxtCommunity = (TextView) findViewById(R.id.main_txt_community);
        this.mTxtMy = (TextView) findViewById(R.id.main_txt_my);
        this.mImgMyDot = (ImageView) findViewById(R.id.main_img_my_dot);
        MeasureUtil.buildNavigationBarHeight(this.mLayoutRoot, false);
        LanguageUtil.setText(this, this.mTxtPlay, "play");
        LanguageUtil.setText(this, this.mTxtStore, "store");
        LanguageUtil.setText(this, this.mTxtCommunity, "community");
        LanguageUtil.setText(this, this.mTxtMy, "me");
        this.mViewPager.setUserInputEnabled(false);
    }

    private void initViewPager() {
        FragmentAdapter2 fragmentAdapter2 = new FragmentAdapter2(this);
        this.mFragmentAdapter = fragmentAdapter2;
        this.mViewPager.setAdapter(fragmentAdapter2);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessage(MessageModel.Message message) {
        try {
            String obj = SharedPreferencesUtil.get(this, Constans.deleteSession).toString();
            if (!TextUtils.isEmpty(obj) && obj.contains("," + message.getUserId() + ",")) {
                SharedPreferencesUtil.put(this, Constans.deleteSession, obj.replace("," + message.getUserId() + ",", ""));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        ContentValues contentValues = new ContentValues();
        contentValues.put("msgId", Integer.valueOf(message.getMsgId()));
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("type", message.getType());
        contentValues.put("voiceLength", Integer.valueOf(message.getVoiceLength()));
        contentValues.put(FirebaseAnalytics.Param.CONTENT, message.getContent());
        contentValues.put("date", message.getDate());
        contentValues.put("revoke", Integer.valueOf(message.getRevoke()));
        contentValues.put("androidRead", "0");
        contentValues.put("referenceId", Integer.valueOf(message.getReferenceId()));
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE, null, contentValues);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageLast(MessageModel.Message message) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("msgLastId", Integer.valueOf(message.getMsgId()));
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_LAST, null, contentValues);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageUser(MessageModel.Message message) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("king", Integer.valueOf(message.getKing()));
        contentValues.put("gradeIcon", message.getGradeIcon());
        contentValues.put("name", message.getName());
        contentValues.put("gender", Integer.valueOf(message.getGender()));
        contentValues.put("age", message.getAge());
        contentValues.put("constellation", message.getConstellation());
        contentValues.put("headIcon", message.getHeadIcon());
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_USER, null, contentValues);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBReject(String str, String str2) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("reject", str2);
        this.dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "msgId = ?", new String[]{str});
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBRevoke(String str) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("revoke", "1");
        this.dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "msgId = ?", new String[]{str});
        this.dbWrite.delete(DatabaseHelper.TABLE_MESSAGE_REFERENCE, "referenceId = ? OR msgId = ?", new String[]{str, str});
    }

    static /* synthetic */ void lambda$requestNotificationPermission$0(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
        }
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_USER_INFO_CHANGE);
        intentFilter.addAction(Constans.RECEIVER_BLE_CONN_STATE);
        intentFilter.addAction(Constans.RECEIVER_BLE_CODE_STATE);
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_READ);
        intentFilter.addAction(Constans.RECEIVER_COMMUNITY_HIDE);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void messageNew(int i) {
        this.baseRequest.getApi().messageNew_v1((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<MessageModel>(this, null, false) { // from class: com.hismith.hismithapp.MainActivity.12
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                if (System.currentTimeMillis() - MainActivity.this.lastMessageTime < ExoPlayer.DEFAULT_DETACH_SURFACE_TIMEOUT_MS || MainActivity.this.mHandler.hasMessages(2)) {
                    MainActivity.this.mHandler.removeMessages(2);
                }
                if (MainActivity.this.isBackground) {
                    return;
                }
                MainActivity.this.mHandler.sendEmptyMessageDelayed(2, MainActivity.this.messageDelayed);
                MainActivity.this.lastMessageTime = System.currentTimeMillis();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                if (System.currentTimeMillis() - MainActivity.this.lastMessageTime < ExoPlayer.DEFAULT_DETACH_SURFACE_TIMEOUT_MS || MainActivity.this.mHandler.hasMessages(2)) {
                    MainActivity.this.mHandler.removeMessages(2);
                }
                if (MainActivity.this.isBackground) {
                    return;
                }
                MainActivity.this.mHandler.sendEmptyMessageDelayed(2, MainActivity.this.messageDelayed);
                MainActivity.this.lastMessageTime = System.currentTimeMillis();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(MessageModel messageModel) {
                int i2;
                if (messageModel.getSuccess() == 1) {
                    if (messageModel.getMessage() != null) {
                        i2 = messageModel.getMessage().length;
                        for (int i3 = 0; i3 < messageModel.getMessage().length; i3++) {
                            MessageModel.Message message = messageModel.getMessage()[i3];
                            if (MyApplication.getApplication().getmUserModel() != null) {
                                if (message.getReferenceMsg() != null) {
                                    message.setReferenceId(message.getReferenceMsg().getReferenceId());
                                    int sendByMessageId = DBMessageUtil.getSendByMessageId(message.getReferenceMsg().getReferenceId());
                                    if (sendByMessageId == 1) {
                                        message.getReferenceMsg().setUserId(MyApplication.getApplication().getmUserModel().getUserId());
                                        message.getReferenceMsg().setUserName(MyApplication.getApplication().getmUserModel().getName());
                                        DBMessageUtil.insertMessageReference(message.getMsgId(), message.getReferenceMsg());
                                    } else if (sendByMessageId == 0) {
                                        message.getReferenceMsg().setUserId(message.getUserId());
                                        message.getReferenceMsg().setUserName(message.getName());
                                        DBMessageUtil.insertMessageReference(message.getMsgId(), message.getReferenceMsg());
                                    }
                                }
                                MainActivity.this.insertDBMessage(message);
                                MainActivity.this.insertDBMessageUser(message);
                                MainActivity.this.insertDBMessageLast(message);
                            }
                            if (message.getRevoke() == 0 && Constans.MESSAGE_TYPE_REMOTE.equals(message.getType())) {
                                try {
                                    Date parse = MainActivity.this.sdf.parse(message.getDate());
                                    if (parse != null) {
                                        LogUtil.e("消息时间：" + parse.toString() + ", 和现在相差" + (System.currentTimeMillis() - parse.getTime()));
                                        if (System.currentTimeMillis() - parse.getTime() < 60000 && AppManager.getInstance().getActivity(ControlAcceptActivity2.class) == null) {
                                            HuDongPlayService.startService(MainActivity.this, 1, message.getMsgId(), message.getName(), message.getHeadIcon(), message.getContent());
                                        }
                                    }
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                    } else {
                        i2 = 0;
                    }
                    StringBuilder sb = new StringBuilder();
                    StringBuilder sb2 = new StringBuilder();
                    try {
                        if (messageModel.getRevokeIds() != null) {
                            for (int i4 = 0; i4 < messageModel.getRevokeIds().length; i4++) {
                                MainActivity.this.msgAction(Integer.parseInt(messageModel.getRevokeIds()[i4].getMsgId()), 3);
                                boolean equals = Constans.MESSAGE_TYPE_REMOTE.equals(messageModel.getRevokeIds()[i4].getType());
                                MainActivity mainActivity = MainActivity.this;
                                if (equals) {
                                    mainActivity.insertDBReject(messageModel.getRevokeIds()[i4].getMsgId(), "2");
                                } else {
                                    mainActivity.insertDBRevoke(messageModel.getRevokeIds()[i4].getMsgId());
                                }
                                sb.append(messageModel.getRevokeIds()[i4].getMsgId());
                                if (i4 < messageModel.getRevokeIds().length - 1) {
                                    sb.append(",");
                                }
                            }
                        }
                    } catch (Exception e2) {
                        e2.printStackTrace();
                    }
                    try {
                        if (messageModel.getRejectIds() != null) {
                            for (int i5 = 0; i5 < messageModel.getRejectIds().length; i5++) {
                                MainActivity.this.msgAction(Integer.parseInt(messageModel.getRejectIds()[i5].getMsgId()), 5);
                                MainActivity.this.insertDBReject(messageModel.getRejectIds()[i5].getMsgId(), "1");
                                sb2.append(messageModel.getRejectIds()[i5].getMsgId());
                                if (i5 < messageModel.getRejectIds().length - 1) {
                                    sb2.append(",");
                                }
                            }
                        }
                    } catch (Exception e3) {
                        e3.printStackTrace();
                    }
                    int messageUnReadCount = DBMessageUtil.getMessageUnReadCount();
                    int interactiveCount = messageModel.getInteractiveCount();
                    MainActivity.this.lastInteractive = interactiveCount;
                    int i6 = MainActivity.this.lastInteractive + messageUnReadCount;
                    MainActivity mainActivity2 = MainActivity.this;
                    if (i6 > 0) {
                        mainActivity2.mImgMyDot.setVisibility(0);
                    } else {
                        mainActivity2.mImgMyDot.setVisibility(8);
                    }
                    LogUtil.w("消息 Message:newMessage:" + i2 + ", unReadCount:" + messageUnReadCount + ", newFirstChat:0. newInteractive:" + interactiveCount + ", newNotice:0");
                    Intent intent = new Intent(Constans.RECEIVER_MESSAGE_NEW);
                    intent.putExtra("revokeId", sb.toString());
                    intent.putExtra("rejectId", sb2.toString());
                    intent.putExtra("newMessage", i2);
                    intent.putExtra("newMessageTotal", messageUnReadCount);
                    intent.putExtra("newFirstChat", 0);
                    intent.putExtra("newInteractive", interactiveCount);
                    intent.putExtra("newNotice", 0);
                    MainActivity.this.sendBroadcast(intent);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void msgAction(int i, int i2) {
        String str;
        String str2 = (String) SharedPreferencesUtil.get(this, Constans.Token);
        if (i2 == 1) {
            str = "delete";
        } else if (i2 == 2) {
            str = "revoke";
        } else if (i2 == 3) {
            str = "revoked";
        } else if (i2 == 4) {
            str = "rejectRP";
        } else if (i2 != 5) {
            return;
        } else {
            str = "rejected";
        }
        this.baseRequest.getApi().msgAction(str2, i, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, null, false) { // from class: com.hismith.hismithapp.MainActivity.13
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                try {
                    httpModel.getSuccess();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33) {
            Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.POST_NOTIFICATIONS").subscribe(new Consumer() { // from class: com.hismith.hismithapp.MainActivity$$ExternalSyntheticLambda0
                @Override // io.reactivex.functions.Consumer
                public final void accept(Object obj) {
                    MainActivity.lambda$requestNotificationPermission$0((Permission) obj);
                }
            });
            if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
                return;
            }
            this.mCompositeDisposable.add(subscribe);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void selectTab(int i) {
        if (i == 1 && this.isDeviceExpired) {
            i = 3;
        }
        if (i == 2 && !this.isShowStore) {
            i = 3;
        }
        if (i == 0) {
            this.mImgPlay.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_play_select));
            this.mImgStore.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_store_unselect));
            this.mImgCommunity.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_community_unselect));
            this.mImgMy.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_my_unselect));
            this.mTxtPlay.setTextColor(ContextCompat.getColor(this, R.color.app_style_color));
            this.mTxtStore.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtCommunity.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtMy.setTextColor(ContextCompat.getColor(this, R.color.gray));
        } else if (i == 1) {
            this.mImgPlay.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_play_unselect));
            this.mImgStore.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_store_unselect));
            this.mImgCommunity.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_community_select));
            this.mImgMy.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_my_unselect));
            this.mTxtPlay.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtStore.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtCommunity.setTextColor(ContextCompat.getColor(this, R.color.app_style_color));
            this.mTxtMy.setTextColor(ContextCompat.getColor(this, R.color.gray));
        } else if (i == 2) {
            this.mImgPlay.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_play_unselect));
            this.mImgStore.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_store_select));
            this.mImgCommunity.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_community_unselect));
            this.mImgMy.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_my_unselect));
            this.mTxtPlay.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtStore.setTextColor(ContextCompat.getColor(this, R.color.app_style_color));
            this.mTxtCommunity.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtMy.setTextColor(ContextCompat.getColor(this, R.color.gray));
        } else if (i == 3) {
            this.mImgPlay.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_play_unselect));
            this.mImgStore.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_store_unselect));
            this.mImgCommunity.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_community_unselect));
            this.mImgMy.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_main_my_select));
            this.mTxtPlay.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtStore.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtCommunity.setTextColor(ContextCompat.getColor(this, R.color.gray));
            this.mTxtMy.setTextColor(ContextCompat.getColor(this, R.color.app_style_color));
        }
        if (i == 2 || this.menuHeight == 0 || this.mLayoutMenu.getHeight() >= this.menuHeight) {
            return;
        }
        ValueAnimator valueAnimator = this.mHideValueAnimator;
        if (valueAnimator != null) {
            valueAnimator.cancel();
        }
        showMenu();
    }

    private void setListener() {
        this.mLayoutPlay.setOnClickListener(this);
        this.mLayoutCommunity.setOnClickListener(this);
        this.mLayoutStore.setOnClickListener(this);
        this.mLayoutMy.setOnClickListener(this);
        this.mViewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() { // from class: com.hismith.hismithapp.MainActivity.3
            @Override // androidx.viewpager2.widget.ViewPager2.OnPageChangeCallback
            public void onPageSelected(int i) {
                super.onPageSelected(i);
                MainActivity.this.selectTab(i);
            }
        });
        if (AppBackgroundUtil.observeSize() == 0) {
            AppBackgroundUtil.observe(new AppBackgroundUtil.Observer() { // from class: com.hismith.hismithapp.MainActivity.4
                @Override // com.hismith.hismithapp.observer.AppBackgroundUtil.Observer
                public void onChange(boolean z) {
                    MainActivity.this.isBackground = z;
                    if (z) {
                        LogUtil.e("AppBackgroundUtil 后台");
                        BleManager.getInstance(MainActivity.this).stopLeScan();
                        if (MainActivity.this.mHandler.hasMessages(2)) {
                            MainActivity.this.mHandler.removeMessages(2);
                            return;
                        }
                        return;
                    }
                    LogUtil.e("AppBackgroundUtil 前台");
                    BleManager.getInstance(MainActivity.this).startLeScan();
                    if (MainActivity.this.mHandler.hasMessages(2)) {
                        return;
                    }
                    MainActivity.this.mHandler.sendEmptyMessage(2);
                }
            });
        }
    }

    private void showMenu() {
        ValueAnimator ofInt = ValueAnimator.ofInt(0, this.menuHeight);
        this.mShowValueAnimator = ofInt;
        ofInt.setInterpolator(new LinearInterpolator());
        this.mShowValueAnimator.addListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.MainActivity.7
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                MainActivity.this.mLayoutMenu.getLayoutParams().height = MainActivity.this.menuHeight;
                MainActivity.this.mLayoutMenu.requestLayout();
            }

            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
            }
        });
        this.mShowValueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.MainActivity.8
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                MainActivity.this.mLayoutMenu.getLayoutParams().height = ((Integer) valueAnimator.getAnimatedValue()).intValue();
                MainActivity.this.mLayoutMenu.requestLayout();
            }
        });
        this.mShowValueAnimator.setDuration(300L);
        this.mShowValueAnimator.start();
        this.isHide = false;
        LogUtil.e("开始显示");
    }

    private void topicCategories() {
        this.baseRequest.getApi().topicCategories((String) SharedPreferencesUtil.get(this, Constans.Token)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicCategoriesModel>(this, null, false) { // from class: com.hismith.hismithapp.MainActivity.14
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicCategoriesModel topicCategoriesModel) {
                if (topicCategoriesModel.getSuccess() == 1) {
                    try {
                        String json = MainActivity.this.gson.toJson(topicCategoriesModel);
                        LogUtil.e("topicCategories jsonStr:" + json);
                        SharedPreferencesUtil.put(MainActivity.this, Constans.TopicCategpries, json);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    MainActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_TOPIC_MENU));
                }
            }
        });
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void changeMenuHeight(int i) {
        LogUtil.e("changeMenuHeight scrollY:" + i);
        int i2 = this.menuHeight;
        if (i2 == 0) {
            i2 = this.mLayoutMenu.getHeight();
            this.menuHeight = i2;
        }
        if (this.isShowStore) {
            LinearLayout linearLayout = this.mLayoutMenu;
            if (i >= i2) {
                if (linearLayout.getHeight() == this.menuHeight && !this.isHide && this.mViewPager.getCurrentItem() == 2) {
                    ValueAnimator ofInt = ValueAnimator.ofInt(this.menuHeight, 0);
                    this.mHideValueAnimator = ofInt;
                    ofInt.setInterpolator(new LinearInterpolator());
                    this.mHideValueAnimator.addListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.MainActivity.5
                        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
                        public void onAnimationEnd(Animator animator) {
                            MainActivity.this.mLayoutMenu.getLayoutParams().height = 0;
                            MainActivity.this.mLayoutMenu.requestLayout();
                        }

                        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
                        public void onAnimationStart(Animator animator) {
                        }
                    });
                    this.mHideValueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.MainActivity.6
                        @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                        public void onAnimationUpdate(ValueAnimator valueAnimator) {
                            int intValue = ((Integer) valueAnimator.getAnimatedValue()).intValue();
                            LogUtil.e("onAnimationUpdate:" + intValue);
                            MainActivity.this.mLayoutMenu.getLayoutParams().height = intValue;
                            MainActivity.this.mLayoutMenu.requestLayout();
                        }
                    });
                    this.mHideValueAnimator.setDuration(300L);
                    this.mHideValueAnimator.start();
                    this.isHide = true;
                    LogUtil.e("开始隐藏");
                }
            } else if (linearLayout.getHeight() == 0 && this.isHide) {
                showMenu();
            }
            LogUtil.e("changeMenuHeight:" + this.mLayoutMenu.getHeight());
        }
    }

    public void complaintReasons() {
        this.baseRequest.getApi().complaintReasons((String) SharedPreferencesUtil.get(this, Constans.Token)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ComplaintReasonsModel>(this, null, false) { // from class: com.hismith.hismithapp.MainActivity.15
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ComplaintReasonsModel complaintReasonsModel) {
                if (complaintReasonsModel.getSuccess() == 1 && complaintReasonsModel.getComplaintReasons() != null && complaintReasonsModel.getComplaintReasons().isJsonObject()) {
                    SharedPreferencesUtil.put(MainActivity.this, Constans.complaintReasons, complaintReasonsModel.getComplaintReasons().getAsJsonObject().toString());
                    for (String str : complaintReasonsModel.getComplaintReasons().getAsJsonObject().keySet()) {
                        LogUtil.e("举报理由：key:" + str + ", value:" + complaintReasonsModel.getComplaintReasons().getAsJsonObject().get(str).getAsString());
                    }
                }
            }
        });
    }

    /* renamed from: lambda$initBluetoothPermission$1$com-hismith-hismithapp-MainActivity, reason: not valid java name */
    /* synthetic */ void m563xac8a8fd8(Permission permission) throws Exception {
        if (permission.granted) {
            BleManager.getInstance(this).startLeScan();
        } else {
            boolean z = permission.shouldShowRequestPermissionRationale;
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.main_layout_play) {
            this.mViewPager.setCurrentItem(0);
            return;
        }
        if (id == R.id.main_layout_store) {
            this.mViewPager.setCurrentItem(2);
            return;
        }
        if (id == R.id.main_layout_community) {
            this.mViewPager.setCurrentItem(1);
            return;
        }
        if (id == R.id.main_layout_my) {
            if (this.isShowStore) {
                this.mViewPager.setCurrentItem(3);
                return;
            }
            boolean z = this.isDeviceExpired;
            ViewPager2 viewPager2 = this.mViewPager;
            if (z) {
                viewPager2.setCurrentItem(1);
            } else {
                viewPager2.setCurrentItem(2);
            }
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(int i) {
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.MainActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                boolean onBackPressed = (MainActivity.this.isShowStore && MainActivity.this.mViewPager.getCurrentItem() == 2) ? MainActivity.this.mShopFragment.onBackPressed() : false;
                if (!onBackPressed && MainActivity.this.mViewPager.getCurrentItem() != 0) {
                    MainActivity.this.mViewPager.setCurrentItem(0);
                    onBackPressed = true;
                }
                if (onBackPressed) {
                    return;
                }
                MainActivity.this.finish();
                if (MainActivity.this.appManager.getActivity(RemotePlayActivity.class) != null) {
                    MainActivity.this.appManager.finishActivity(RemotePlayActivity.class);
                }
            }
        });
        setContentView(R.layout.activity_main);
        initView();
        initValue(bundle);
        setListener();
        UpgradeManager.initUpgrade(this);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        LogUtil.e("注销过程 MainActivity onDestory()");
        AppBackgroundUtil.clearObserver();
        this.mHandler.removeCallbacksAndMessages(null);
        unRegisterReceiver();
        colseSocket();
        RemoteFloatingManager.getInstance().destroyFloatWindow();
        closeDB();
        BleManager.getInstance(this).writeShutdown();
        BleManager.getInstance(this).stopLeScan();
        BleManager.getInstance(this).release();
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(int i) {
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
        BleManager.getInstance(this).setOnConnectStateListener(null);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
        if (this.appManager.getActivity(RemotePlayActivity.class) != null && (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) == null || !FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).isShowing())) {
            startActivity(new Intent(this, (Class<?>) RemotePlayActivity.class));
        }
        BleManager.getInstance(this).setOnConnectStateListener(this);
    }
}
