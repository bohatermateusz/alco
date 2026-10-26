package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;
import com.facebook.appevents.integrity.IntegrityManager;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.BluetoothMultitudeFragment;
import com.hismith.hismithapp.fragment.DeviceUIListFragment;
import com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment;
import com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment;
import com.hismith.hismithapp.fragment.playui.SMPlayFragment;
import com.hismith.hismithapp.listener.CustomClickListener;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.RightDraggableFrameLayout;
import com.hismith.hismithapp.view.plug.PlugPiuTimer;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;

/* loaded from: classes3.dex */
public class IPlayOffActivity extends BaseActivity implements BleManager.OnConnectStateListener, BluetoothMultitudeFragment.FragmentInteractionListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private FrameLayout frame_multitude;
    private ImageView img_bluetooth;
    private RightDraggableFrameLayout layout_fragment_piu;
    private ActivityResultLauncher mBluetoothLauncher;
    private DeviceUIListFragment mDeviceUIListFragment;
    private Fragment mFragment;
    private ImageView mImgCloseLeft;
    private ImageView mImgCloseRight;
    private ImageView mImgGestureLeft;
    private ImageView mImgGestureRight;
    private BluetoothMultitudeFragment mMultitudeFragment;
    private PlugPiuTimer mPlugPiuTimer;
    private TextView mTxtGestureTip;
    private View mViewTouch;
    private CustomWebView mWebView;
    private TextView title_txt_status;
    private int leftAction = 1;
    private int rightAction = 1;
    private String mMultitudeFragmentTag = "fragment_multitude_tag";
    private boolean isMultitudeFragmentVisible = false;
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayOffActivity.4
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayOffActivity.this.leftAction = 0;
                IPlayOffActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayOffActivity.this.lastLeftClickTime;
                IPlayOffActivity iPlayOffActivity = IPlayOffActivity.this;
                if (j < 500) {
                    iPlayOffActivity.finish();
                    return true;
                }
                iPlayOffActivity.lastLeftClickTime = currentTimeMillis;
                IPlayOffActivity.this.leftAction = 1;
                IPlayOffActivity.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayOffActivity.5
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayOffActivity.this.rightAction = 0;
                IPlayOffActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayOffActivity.this.lastRightClickTime;
                IPlayOffActivity iPlayOffActivity = IPlayOffActivity.this;
                if (j < 500) {
                    iPlayOffActivity.finish();
                    return true;
                }
                iPlayOffActivity.lastRightClickTime = currentTimeMillis;
                IPlayOffActivity.this.rightAction = 1;
            }
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.IPlayOffActivity.6
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                IPlayOffActivity.this.initFragment();
                return;
            }
            if (Constans.RECEIVER_BLE_DEVICE_STATE.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.EXTRA_DEVICE_ADDRESS);
                int intExtra = intent.getIntExtra("ble_state", 0);
                IPlayOffActivity iPlayOffActivity = IPlayOffActivity.this;
                if (intExtra != 2) {
                    iPlayOffActivity.initFragment();
                    if (BleManager.getInstance(IPlayOffActivity.this).hasPiuFunBle()) {
                        return;
                    }
                    IPlayOffActivity.this.removePlugPiuTimerFragment();
                    return;
                }
                boolean isFType = ProductUtil.isFType(iPlayOffActivity, stringExtra);
                IPlayOffActivity iPlayOffActivity2 = IPlayOffActivity.this;
                if (!isFType) {
                    iPlayOffActivity2.initFragment();
                    return;
                } else {
                    if (BleManager.getInstance(iPlayOffActivity2).hasPiuFunBle()) {
                        IPlayOffActivity.this.plugPiuTimerFragment();
                        return;
                    }
                    return;
                }
            }
            if (action.equals(Constans.RECEIVER_BLE_CONN_STATE)) {
                int intExtra2 = intent.getIntExtra("ble_state", 0);
                IPlayOffActivity iPlayOffActivity3 = IPlayOffActivity.this;
                if (intExtra2 == 2) {
                    iPlayOffActivity3.title_txt_status.setVisibility(8);
                    IPlayOffActivity.this.img_bluetooth.setImageDrawable(ContextCompat.getDrawable(IPlayOffActivity.this, R.mipmap.ic_play_bluetooth_connect));
                    return;
                } else {
                    iPlayOffActivity3.title_txt_status.setVisibility(0);
                    IPlayOffActivity.this.img_bluetooth.setImageDrawable(ContextCompat.getDrawable(IPlayOffActivity.this, R.mipmap.ic_play_bluetooth_disconnect));
                    return;
                }
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                LogUtil.e("广播RECEIVER_BLE_RECE_DATA IPlayOffActivity");
                String stringExtra2 = intent.getStringExtra(Constans.BleType);
                String stringExtra3 = intent.getStringExtra(Constans.BleData0);
                String stringExtra4 = intent.getStringExtra(Constans.BleData1);
                String stringExtra5 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                if (IPlayOffActivity.this.mDeviceUIListFragment != null) {
                    IPlayOffActivity.this.mDeviceUIListFragment.buildBleReceData(stringExtra5, stringExtra2, stringExtra3, stringExtra4);
                }
            }
        }
    };

    private void addMultitudeFragment() {
        BluetoothMultitudeFragment bluetoothMultitudeFragment = this.mMultitudeFragment;
        if (bluetoothMultitudeFragment == null || !bluetoothMultitudeFragment.isAdded()) {
            FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
            if (this.frame_multitude.getVisibility() == 8) {
                this.frame_multitude.setVisibility(0);
            }
            this.mMultitudeFragment = new BluetoothMultitudeFragment();
            beginTransaction.replace(R.id.frame_multitude, this.mMultitudeFragment, this.mMultitudeFragmentTag);
            beginTransaction.commit();
            this.isMultitudeFragmentVisible = true;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean bluetoothIsEnabled() {
        if (BleManager.getInstance(this).getBluetoothAdapter() == null) {
            return false;
        }
        if (BleManager.getInstance(this).getBluetoothAdapter().isEnabled()) {
            return true;
        }
        Intent intent = new Intent("android.bluetooth.adapter.action.REQUEST_ENABLE");
        ActivityResultLauncher activityResultLauncher = this.mBluetoothLauncher;
        if (activityResultLauncher == null) {
            return false;
        }
        activityResultLauncher.launch(intent);
        return false;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exitIPlay() {
        if (this.leftAction == 0 && this.rightAction == 0) {
            finish();
        }
        if (this.leftAction == 1) {
            LogUtil.e("onTouch 左抖动");
            shakeAnimation(this.mImgCloseLeft);
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            LogUtil.e("onTouch 右抖动");
            shakeAnimation(this.mImgCloseRight);
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "exitDoubleGameNote"));
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideGestureTip() {
        if (this.mImgGestureLeft.getVisibility() == 0) {
            this.mImgGestureLeft.clearAnimation();
            this.mImgGestureLeft.setVisibility(8);
        }
        if (this.mImgGestureRight.getVisibility() == 0) {
            this.mImgGestureRight.clearAnimation();
            this.mImgGestureRight.setVisibility(8);
        }
        if (this.mTxtGestureTip.getVisibility() == 0) {
            this.mTxtGestureTip.setVisibility(8);
        }
    }

    private void init() {
        if (BleManager.getInstance(this).getStatus() == 2) {
            this.img_bluetooth.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_connect));
        }
        initBluetoothPermission();
        initFragment();
        initActivityResult();
        registerReceiver();
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.IPlayOffActivity.7
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                if (activityResult.getResultCode() == -1) {
                    LogUtil.e("蓝牙打开");
                    IPlayOffActivity.this.startActivity(new Intent(IPlayOffActivity.this, (Class<?>) BluetootListActivity.class));
                } else if (activityResult.getResultCode() == 0) {
                    LogUtil.e("蓝牙打不开");
                }
            }
        });
    }

    private void initBluetoothPermission() {
        Disposable subscribe = new RxPermissions(this).requestEachCombined(Build.VERSION.SDK_INT >= 31 ? new String[]{"android.permission.BLUETOOTH_SCAN", "android.permission.BLUETOOTH_CONNECT"} : new String[]{"android.permission.ACCESS_COARSE_LOCATION", "android.permission.ACCESS_FINE_LOCATION"}).subscribe(new Consumer() { // from class: com.hismith.hismithapp.IPlayOffActivity$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                IPlayOffActivity.this.m561xf5860b10((Permission) obj);
            }
        });
        if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initFragment() {
        Fragment cC_SMSPlayFragment = ProductUtil.isSMSType() ? new CC_SMSPlayFragment() : ProductUtil.isSMVType() ? new CC_SMVPlayFragment() : !ProductUtil.isFType() ? new SMPlayFragment() : null;
        if (cC_SMSPlayFragment != null) {
            LogUtil.e("initFragment fragment :" + cC_SMSPlayFragment.getClass().getSimpleName());
            Fragment fragment = this.mFragment;
            if (fragment == null || fragment.getClass() != cC_SMSPlayFragment.getClass()) {
                this.mFragment = cC_SMSPlayFragment;
            }
            FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
            beginTransaction.replace(R.id.frame_layout, this.mFragment);
            beginTransaction.commitAllowingStateLoss();
        } else if (this.mFragment != null) {
            LogUtil.e("initFragment removemFragment :" + this.mFragment.getClass().getSimpleName());
            FragmentTransaction beginTransaction2 = getSupportFragmentManager().beginTransaction();
            beginTransaction2.remove(this.mFragment);
            beginTransaction2.commit();
            this.mFragment = null;
        }
        if (BleManager.getInstance(this).hasPiuFunBle()) {
            plugPiuTimerFragment();
        } else {
            removePlugPiuTimerFragment();
        }
        plugDeviceUIListFragment();
    }

    private void initView() {
        this.title_txt_status = (TextView) findViewById(R.id.title_txt_status);
        this.img_bluetooth = (ImageView) findViewById(R.id.img_bluetooth);
        this.frame_multitude = (FrameLayout) findViewById(R.id.frame_multitude);
        this.layout_fragment_piu = (RightDraggableFrameLayout) findViewById(R.id.layout_fragment_piu);
        this.mViewTouch = findViewById(R.id.view_touch);
        this.mImgCloseLeft = (ImageView) findViewById(R.id.iplay_img_close_left);
        this.mImgCloseRight = (ImageView) findViewById(R.id.iplay_img_close_right);
        this.mImgGestureLeft = (ImageView) findViewById(R.id.iplay_img_gesture_left);
        this.mImgGestureRight = (ImageView) findViewById(R.id.iplay_img_gesture_right);
        this.mTxtGestureTip = (TextView) findViewById(R.id.iplay_txt_gesture_tip);
        this.mDeviceUIListFragment = (DeviceUIListFragment) getSupportFragmentManager().findFragmentById(R.id.fragment_device_list);
        this.layout_fragment_piu.setTopBoundViewId(R.id.img_bluetooth);
        this.layout_fragment_piu.setBottomBoundViewId(R.id.iplay_img_close_right);
        LanguageUtil.setText(this, this.title_txt_status, "bluetoothDisconnected");
        LanguageUtil.setText(this, this.mTxtGestureTip, "exitDoubleGameNote");
        AnimUtil.translate(this, this.mImgGestureLeft);
        AnimUtil.translate(this, this.mImgGestureRight);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void plugDeviceUIListFragment() {
        DeviceUIListFragment deviceUIListFragment = this.mDeviceUIListFragment;
        if (deviceUIListFragment == null) {
            deviceUIListFragment = (DeviceUIListFragment) getSupportFragmentManager().findFragmentById(R.id.fragment_device_list);
            this.mDeviceUIListFragment = deviceUIListFragment;
        }
        if (deviceUIListFragment != null) {
            deviceUIListFragment.refreshDeviceInfo();
            boolean isEmpty = this.mDeviceUIListFragment.getmList().isEmpty();
            DeviceUIListFragment deviceUIListFragment2 = this.mDeviceUIListFragment;
            if (isEmpty) {
                if (deviceUIListFragment2.getView() != null) {
                    this.mDeviceUIListFragment.getView().setVisibility(8);
                }
            } else {
                if (deviceUIListFragment2.getView() != null) {
                    this.mDeviceUIListFragment.getView().setVisibility(0);
                }
                this.mDeviceUIListFragment.refreshDeviceInfo();
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void plugPiuTimerFragment() {
        LogUtil.e("initFragment plugPiuTimerFragment :");
        FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
        this.mPlugPiuTimer = new PlugPiuTimer();
        beginTransaction.replace(R.id.layout_fragment_piu, this.mPlugPiuTimer);
        beginTransaction.commitAllowingStateLoss();
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_CONN_STATE);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void removeMultitudeFragment() {
        BluetoothMultitudeFragment bluetoothMultitudeFragment = this.mMultitudeFragment;
        if (bluetoothMultitudeFragment == null || !bluetoothMultitudeFragment.isAdded()) {
            return;
        }
        FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
        beginTransaction.remove(this.mMultitudeFragment);
        beginTransaction.commit();
        if (this.frame_multitude.getVisibility() == 0) {
            this.frame_multitude.setVisibility(8);
        }
        this.mMultitudeFragment = null;
        this.isMultitudeFragmentVisible = false;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void removePlugPiuTimerFragment() {
        LogUtil.e("initFragment removePlugPiuTimerFragment :");
        if (this.mPlugPiuTimer != null) {
            FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
            beginTransaction.remove(this.mPlugPiuTimer);
            beginTransaction.commitAllowingStateLoss();
            this.mPlugPiuTimer = null;
        }
    }

    private void setListener() {
        BleManager.getInstance(this).setOnConnectStateListener(this);
        this.img_bluetooth.setOnClickListener(new CustomClickListener() { // from class: com.hismith.hismithapp.IPlayOffActivity.2
            @Override // com.hismith.hismithapp.listener.CustomClickListener
            protected void onSingleClick(View view) {
                if (IPlayOffActivity.this.bluetoothIsEnabled()) {
                    IPlayOffActivity.this.toggleMultitudeFragment();
                }
            }
        });
        this.mViewTouch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayOffActivity.3
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                IPlayOffActivity.this.hideGestureTip();
                return false;
            }
        });
        this.mImgCloseLeft.setOnTouchListener(this.leftOnTouch);
        this.mImgCloseRight.setOnTouchListener(this.rightOnTouch);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void toggleMultitudeFragment() {
        if (this.isMultitudeFragmentVisible) {
            removeMultitudeFragment();
        } else {
            addMultitudeFragment();
        }
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$initBluetoothPermission$0$com-hismith-hismithapp-IPlayOffActivity, reason: not valid java name */
    /* synthetic */ void m561xf5860b10(Permission permission) throws Exception {
        if (permission.granted) {
            BleManager.getInstance(this).startLeScan();
        } else {
            boolean z = permission.shouldShowRequestPermissionRationale;
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(final int i) {
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayOffActivity.8
            @Override // java.lang.Runnable
            public void run() {
                if (IPlayOffActivity.this.mMultitudeFragment != null) {
                    IPlayOffActivity.this.mMultitudeFragment.onConnected(i);
                }
                IPlayOffActivity.this.plugDeviceUIListFragment();
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.IPlayOffActivity.1
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                if (IPlayOffActivity.this.isMultitudeFragmentVisible) {
                    IPlayOffActivity.this.toggleMultitudeFragment();
                }
            }
        });
        getWindow().addFlags(128);
        setContentView(R.layout.activity_iplay_off);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        unRegisterReceiver();
        ControlRuleManager.getInstance().removeHandleCallback();
        BleManager.getInstance(this).stopLeScan();
        BleManager.getInstance(this).release();
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(final int i) {
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayOffActivity.9
            @Override // java.lang.Runnable
            public void run() {
                if (IPlayOffActivity.this.mMultitudeFragment != null) {
                    IPlayOffActivity.this.mMultitudeFragment.onDisconnected(i);
                }
                IPlayOffActivity.this.plugDeviceUIListFragment();
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.FragmentInteractionListener
    public void onFragmentRemoved(String str) {
        this.isMultitudeFragmentVisible = false;
    }

    public void shakeAnimation(View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(this, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }
}
