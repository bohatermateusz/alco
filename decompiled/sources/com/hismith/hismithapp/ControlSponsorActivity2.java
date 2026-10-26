package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.IBinder;
import android.text.TextUtils;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentTransaction;
import com.bumptech.glide.Glide;
import com.bumptech.glide.request.BaseRequestOptions;
import com.bumptech.glide.request.RequestOptions;
import com.google.gson.Gson;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.dialog.HudongSettingDialog;
import com.hismith.hismithapp.fragment.PlaySMSFragment2;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.service.HuDongPlayService;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CheckUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.StatusBarUtil;
import com.hismith.hismithapp.view.CircleImageView;
import jp.wasabeef.glide.transformations.BlurTransformation;

/* loaded from: classes3.dex */
public class ControlSponsorActivity2 extends BaseActivity implements View.OnClickListener, BleManager.OnConnectStateListener {
    private RetrofitRequest baseRequest;
    private HudongSettingDialog hudongSettingDialog;
    private ImageView img_head;
    private CircleImageView img_logo;
    private ImageView img_refuse;
    private ImageView img_setting;
    private ImageView img_small;
    private ImageView iplay_layout_bg;
    private ConstraintLayout layout;
    private FrameLayout layout_bg;
    private ConstraintLayout layout_content;
    private LinearLayout layout_control;
    private FrameLayout layout_frame;
    private LinearLayout layout_refuse;
    private FrameLayout layout_root;
    private HuDongPlayService.HuDongBinder mBinder;
    private PlaySMSFragment2 mFragment;
    private Gson mGson;
    private TextView title_txt_name;
    private TextView title_txt_status;
    private TextView txt_refuse;
    private TextView txt_remote;
    private final int REQUEST_SMALL = 101;
    private ServiceConnection serviceConnection = new ServiceConnection() { // from class: com.hismith.hismithapp.ControlSponsorActivity2.1
        @Override // android.content.ServiceConnection
        public void onServiceConnected(ComponentName componentName, IBinder iBinder) {
            ControlSponsorActivity2.this.mBinder = (HuDongPlayService.HuDongBinder) iBinder;
            ControlSponsorActivity2 controlSponsorActivity2 = ControlSponsorActivity2.this;
            LanguageUtil.setText(controlSponsorActivity2, controlSponsorActivity2.title_txt_name, ControlSponsorActivity2.this.mBinder.getName());
            LogUtil.e("发起者显示头像：" + ControlSponsorActivity2.this.mBinder.getHeadIcon());
            if (!TextUtils.isEmpty(ControlSponsorActivity2.this.mBinder.getHeadIcon())) {
                Glide.with((FragmentActivity) ControlSponsorActivity2.this).load(ControlSponsorActivity2.this.mBinder.getHeadIcon()).apply((BaseRequestOptions<?>) RequestOptions.bitmapTransform(new BlurTransformation(90))).into(ControlSponsorActivity2.this.img_head);
            }
            ControlSponsorActivity2.this.mBinder.destroyFloatWindow();
            if (HuDongPlayService.getClientConnection() == 1) {
                ControlSponsorActivity2 controlSponsorActivity22 = ControlSponsorActivity2.this;
                controlSponsorActivity22.initFragment(controlSponsorActivity22.mBinder.getSocketProductModel());
            }
            ControlSponsorActivity2.this.mBinder.setOnServiceNotifyListener(new HuDongPlayService.OnServiceNotifyListener() { // from class: com.hismith.hismithapp.ControlSponsorActivity2.1.1
                @Override // com.hismith.hismithapp.service.HuDongPlayService.OnServiceNotifyListener
                public void onProductChange(SocketProductModel socketProductModel) {
                    ControlSponsorActivity2.this.initFragment(ControlSponsorActivity2.this.mBinder.getSocketProductModel());
                }
            });
        }

        @Override // android.content.ServiceConnection
        public void onServiceDisconnected(ComponentName componentName) {
            ControlSponsorActivity2.this.mBinder.setOnServiceNotifyListener(null);
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.ControlSponsorActivity2.4
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                LogUtil.e("蓝牙广播数据接收：" + intent.getStringExtra(Constans.BleType) + ", data0:" + intent.getStringExtra(Constans.BleData0) + ", data1:" + intent.getStringExtra(Constans.BleData1));
                return;
            }
            if (Constans.RECEIVER_MESSAGE_NEW.equals(action)) {
                String stringExtra = intent.getStringExtra("rejectId");
                if (TextUtils.isEmpty(stringExtra)) {
                    return;
                }
                for (String str : stringExtra.split(",")) {
                    if (str.equals(ControlSponsorActivity2.this.mBinder.getMsgId() + "")) {
                        ControlSponsorActivity2.this.finish();
                    }
                }
                return;
            }
            if (SocketManager.RECEIVER_SOCKET_NOTIFY.equals(action)) {
                int intExtra = intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, -1);
                String stringExtra2 = intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA);
                if (intExtra != 0) {
                    if (intExtra == 1) {
                        ControlSponsorActivity2.this.finish();
                    }
                } else {
                    if (ControlSponsorActivity2.this.mFragment == null || !ControlSponsorActivity2.this.mFragment.isAdded()) {
                        return;
                    }
                    ControlSponsorActivity2.this.mFragment.onNotify(stringExtra2);
                }
            }
        }
    };

    private void checkFloatPermission() {
        if (CheckUtil.checkOp(this, 24) != 0) {
            startActivityForResult(new Intent("android.settings.action.MANAGE_OVERLAY_PERMISSION", Uri.parse("package:" + getPackageName())), 101);
            return;
        }
        HuDongPlayService.HuDongBinder huDongBinder = this.mBinder;
        if (huDongBinder != null) {
            huDongBinder.showFloatWindow();
        }
        finish();
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.mGson = new Gson();
        bindService(new Intent(this, (Class<?>) HuDongPlayService.class), this.serviceConnection, 1);
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initFragment(SocketProductModel socketProductModel) {
        ConstraintLayout constraintLayout = this.layout;
        if (constraintLayout != null) {
            constraintLayout.setVisibility(8);
        }
        FrameLayout frameLayout = this.layout_bg;
        if (frameLayout != null) {
            frameLayout.setVisibility(8);
        }
        LinearLayout linearLayout = this.layout_control;
        if (linearLayout != null) {
            linearLayout.setVisibility(0);
        }
        PlaySMSFragment2 playSMSFragment2 = this.mFragment;
        if (playSMSFragment2 != null) {
            playSMSFragment2.updateSocketProductModel(socketProductModel);
            return;
        }
        FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
        this.mFragment = new PlaySMSFragment2();
        Bundle bundle = new Bundle();
        bundle.putSerializable("ProductModel", socketProductModel);
        this.mFragment.setArguments(bundle);
        beginTransaction.replace(R.id.layout_frame, this.mFragment);
        beginTransaction.commitAllowingStateLoss();
    }

    private void initView() {
        this.layout_root = (FrameLayout) findViewById(R.id.layout_root);
        this.img_head = (ImageView) findViewById(R.id.img_head);
        this.title_txt_name = (TextView) findViewById(R.id.title_txt_name);
        this.txt_remote = (TextView) findViewById(R.id.txt_remote);
        this.img_logo = (CircleImageView) findViewById(R.id.img_logo);
        this.layout_refuse = (LinearLayout) findViewById(R.id.layout_refuse);
        this.img_refuse = (ImageView) findViewById(R.id.img_refuse);
        this.txt_refuse = (TextView) findViewById(R.id.txt_refuse);
        this.layout = (ConstraintLayout) findViewById(R.id.layout);
        this.layout_bg = (FrameLayout) findViewById(R.id.layout_bg);
        this.layout_control = (LinearLayout) findViewById(R.id.layout_control);
        this.layout_content = (ConstraintLayout) findViewById(R.id.layout_content);
        this.iplay_layout_bg = (ImageView) findViewById(R.id.iplay_layout_bg);
        this.title_txt_status = (TextView) findViewById(R.id.title_txt_status);
        this.layout_frame = (FrameLayout) findViewById(R.id.layout_frame);
        this.img_small = (ImageView) findViewById(R.id.img_small);
        this.img_setting = (ImageView) findViewById(R.id.img_setting);
        LanguageUtil.setText(this, this.txt_remote, "remotePlay");
        LanguageUtil.setText(this, this.txt_refuse, "cancel");
        LanguageUtil.setText(this, this.title_txt_status, "bluetoothDisconnected");
        Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.iplay_layout_bg);
        StatusBarUtil.fitTitleBar(this, this.layout);
        MeasureUtil.buildNavigationBarHeight(this.layout, false);
        MeasureUtil.buildNavigationBarHeight(this.layout_control, true);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_NEW);
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.img_refuse.setOnClickListener(this);
        this.img_small.setOnClickListener(this);
        this.img_setting.setOnClickListener(this);
        BleManager.getInstance(this).setOnConnectStateListener(this);
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void exit() {
        HuDongPlayService.HuDongBinder huDongBinder = this.mBinder;
        if (huDongBinder != null) {
            huDongBinder.exit();
        }
    }

    public HuDongPlayService.HuDongBinder getBinder() {
        return this.mBinder;
    }

    public int getStartPositionSafeMax() {
        HuDongPlayService.HuDongBinder huDongBinder = this.mBinder;
        if (huDongBinder != null) {
            return huDongBinder.getmStartPositionSafeMax();
        }
        return 0;
    }

    public int getStopPositionSafeMax() {
        HuDongPlayService.HuDongBinder huDongBinder = this.mBinder;
        if (huDongBinder != null) {
            return huDongBinder.getmStopPositionSafeMax();
        }
        return 100;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.img_refuse) {
            this.img_refuse.setEnabled(false);
            DBMessageUtil.updateDBReject(this, this.mBinder.getMsgId() + "", "2");
            HuDongPlayService.HuDongBinder huDongBinder = this.mBinder;
            huDongBinder.msgAction(huDongBinder.getMsgId(), 2);
            exit();
            return;
        }
        if (view.getId() == R.id.img_small) {
            checkFloatPermission();
            return;
        }
        if (view.getId() == R.id.img_setting) {
            HudongSettingDialog hudongSettingDialog = this.hudongSettingDialog;
            if (hudongSettingDialog != null) {
                hudongSettingDialog.dismiss();
            }
            HuDongPlayService.HuDongBinder huDongBinder2 = this.mBinder;
            int safeMax = huDongBinder2 != null ? huDongBinder2.getSafeMax() : 100;
            HuDongPlayService.HuDongBinder huDongBinder3 = this.mBinder;
            HudongSettingDialog hudongSettingDialog2 = new HudongSettingDialog(this, safeMax, true, huDongBinder3 != null ? huDongBinder3.getVibrateSafeMax() : 100, new HudongSettingDialog.OnSafeMaxListener() { // from class: com.hismith.hismithapp.ControlSponsorActivity2.3
                @Override // com.hismith.hismithapp.dialog.HudongSettingDialog.OnSafeMaxListener
                public void onSafeMax(int i) {
                    if (ControlSponsorActivity2.this.mBinder != null) {
                        ControlSponsorActivity2.this.mBinder.setSafeMax(i);
                        if (ControlSponsorActivity2.this.mFragment == null || !ControlSponsorActivity2.this.mFragment.isAdded()) {
                            return;
                        }
                        ControlSponsorActivity2.this.mFragment.onSetSafeMax(i);
                    }
                }

                @Override // com.hismith.hismithapp.dialog.HudongSettingDialog.OnSafeMaxListener
                public void onVibrateSafeMax(int i) {
                    if (ControlSponsorActivity2.this.mBinder != null) {
                        ControlSponsorActivity2.this.mBinder.setVibrateSafeMax(i);
                        if (ControlSponsorActivity2.this.mFragment == null || !ControlSponsorActivity2.this.mFragment.isAdded()) {
                            return;
                        }
                        ControlSponsorActivity2.this.mFragment.onSetVibrateSafeMax(i);
                    }
                }
            });
            this.hudongSettingDialog = hudongSettingDialog2;
            hudongSettingDialog2.show();
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(int i) {
        LogUtil.e("onConnected size:" + i);
        if (i > 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.ControlSponsorActivity2.5
                @Override // java.lang.Runnable
                public void run() {
                    ControlSponsorActivity2.this.title_txt_status.setVisibility(8);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.ControlSponsorActivity2.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
            }
        });
        setContentView(R.layout.activity_control_sponsor);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        unbindService(this.serviceConnection);
        unRegisterReceiver();
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(int i) {
        LogUtil.e("onDisconnected size:" + i);
        if (i == 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.ControlSponsorActivity2.6
                @Override // java.lang.Runnable
                public void run() {
                    ControlSponsorActivity2.this.title_txt_status.setVisibility(0);
                }
            });
        }
    }

    public void setStartPositionSafeMax(int i) {
        HuDongPlayService.HuDongBinder huDongBinder = this.mBinder;
        if (huDongBinder != null) {
            huDongBinder.setmStartPositionSafeMax(i);
        }
    }

    public void setStopPositionSafeMax(int i) {
        HuDongPlayService.HuDongBinder huDongBinder = this.mBinder;
        if (huDongBinder != null) {
            huDongBinder.setmStopPositionSafeMax(i);
        }
    }
}
