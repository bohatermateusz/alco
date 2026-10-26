package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;
import androidx.media3.exoplayer.ExoPlayer;
import com.facebook.appevents.integrity.IntegrityManager;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.BluetoothMultitudeFragment;
import com.hismith.hismithapp.fragment.DeviceUIListFragment;
import com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment;
import com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment;
import com.hismith.hismithapp.fragment.playui.SMPlayFragment;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.RightDraggableFrameLayout;
import com.hismith.hismithapp.view.plug.PlugPiuTimer;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class PlaySoloActivity extends BaseActivity implements BleManager.OnConnectStateListener, View.OnClickListener, BluetoothMultitudeFragment.FragmentInteractionListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private static final int WHAT_PLAY_HEART = 1;
    private RetrofitRequest baseRequest;
    private RightDraggableFrameLayout draglayout_device_list;
    private FrameLayout frame_layout;
    private FrameLayout frame_multitude;
    private ImageView img_bluetooth_multitude;
    private RightDraggableFrameLayout layout_fragment_piu;
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
    private TextView title_txt_status;
    private int leftAction = 1;
    private int rightAction = 1;
    private String mMultitudeFragmentTag = "fragment_multitude_tag";
    private boolean isMultitudeFragmentVisible = false;
    private String productId = "";
    private final int DELAY_PLAY_HEART = ExoPlayer.DEFAULT_STUCK_PLAYING_NOT_ENDING_TIMEOUT_MS;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.PlaySoloActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1) {
                PlaySoloActivity.this.setPlayLogs(0);
                PlaySoloActivity.this.mHandler.sendEmptyMessageDelayed(1, 60000L);
            }
            return false;
        }
    });
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.PlaySoloActivity.4
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySoloActivity.this.leftAction = 0;
                PlaySoloActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySoloActivity.this.lastLeftClickTime;
                PlaySoloActivity playSoloActivity = PlaySoloActivity.this;
                if (j < 500) {
                    playSoloActivity.finish();
                    return true;
                }
                playSoloActivity.lastLeftClickTime = currentTimeMillis;
                PlaySoloActivity.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.PlaySoloActivity.5
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySoloActivity.this.rightAction = 0;
                PlaySoloActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySoloActivity.this.lastRightClickTime;
                PlaySoloActivity playSoloActivity = PlaySoloActivity.this;
                if (j < 500) {
                    playSoloActivity.finish();
                    return true;
                }
                playSoloActivity.lastRightClickTime = currentTimeMillis;
                PlaySoloActivity.this.rightAction = 1;
            }
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.PlaySoloActivity.7
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                PlaySoloActivity.this.initFragment();
                BleManager.getInstance(PlaySoloActivity.this).initWriteMessage();
            } else if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                LogUtil.e("广播RECEIVER_BLE_RECE_DATA PlaySoloActivity");
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                String stringExtra4 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                if (PlaySoloActivity.this.mDeviceUIListFragment != null) {
                    PlaySoloActivity.this.mDeviceUIListFragment.buildBleReceData(stringExtra4, stringExtra, stringExtra2, stringExtra3);
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
    public void exitIPlay() {
        if (this.leftAction == 0 && this.rightAction == 0) {
            finish();
        }
        if (this.leftAction == 1) {
            shakeAnimation(this.mImgCloseLeft);
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
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
        this.baseRequest = new RetrofitRequest();
        initFragment();
        BleManager.getInstance(this).initWriteMessage();
        registerReceiver();
        this.mHandler.sendEmptyMessage(1);
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
        if (!TextUtils.isEmpty(MyApplication.getApplication().getDeviceCode())) {
            this.productId = MyApplication.getApplication().getDeviceCode();
        }
        plugDeviceUIListFragment();
    }

    private void initView() {
        this.title_txt_status = (TextView) findViewById(R.id.title_txt_status);
        this.img_bluetooth_multitude = (ImageView) findViewById(R.id.img_bluetooth_multitude);
        this.frame_layout = (FrameLayout) findViewById(R.id.frame_layout);
        this.frame_multitude = (FrameLayout) findViewById(R.id.frame_multitude);
        this.layout_fragment_piu = (RightDraggableFrameLayout) findViewById(R.id.layout_fragment_piu);
        this.draglayout_device_list = (RightDraggableFrameLayout) findViewById(R.id.draglayout_device_list);
        this.mViewTouch = findViewById(R.id.view_touch);
        this.mImgCloseLeft = (ImageView) findViewById(R.id.iplay_img_close_left);
        this.mImgCloseRight = (ImageView) findViewById(R.id.iplay_img_close_right);
        this.mImgGestureLeft = (ImageView) findViewById(R.id.iplay_img_gesture_left);
        this.mImgGestureRight = (ImageView) findViewById(R.id.iplay_img_gesture_right);
        this.mTxtGestureTip = (TextView) findViewById(R.id.iplay_txt_gesture_tip);
        this.mDeviceUIListFragment = (DeviceUIListFragment) getSupportFragmentManager().findFragmentById(R.id.fragment_device_list);
        this.layout_fragment_piu.setTopBoundViewId(R.id.img_bluetooth_multitude);
        this.layout_fragment_piu.setBottomBoundViewId(R.id.iplay_img_close_right);
        this.draglayout_device_list.setTopBoundViewId(R.id.img_bluetooth_multitude);
        this.draglayout_device_list.setBottomBoundViewId(R.id.iplay_img_close_right);
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

    private void plugPiuTimerFragment() {
        LogUtil.e("initFragment plugPiuTimerFragment :");
        FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
        this.mPlugPiuTimer = new PlugPiuTimer();
        beginTransaction.replace(R.id.layout_fragment_piu, this.mPlugPiuTimer);
        beginTransaction.commitAllowingStateLoss();
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
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

    private void removePlugPiuTimerFragment() {
        if (this.mPlugPiuTimer != null) {
            FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
            beginTransaction.remove(this.mPlugPiuTimer);
            beginTransaction.commitAllowingStateLoss();
            this.mPlugPiuTimer = null;
        }
    }

    private void setListener() {
        BleManager.getInstance(this).setOnConnectStateListener(this);
        this.img_bluetooth_multitude.setOnClickListener(this);
        this.mViewTouch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.PlaySoloActivity.3
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                PlaySoloActivity.this.hideGestureTip();
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

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (R.id.img_bluetooth_multitude == view.getId()) {
            toggleMultitudeFragment();
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(final int i) {
        LogUtil.e("plugDeviceUIListFragment onConnected size:" + i);
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlaySoloActivity.8
            @Override // java.lang.Runnable
            public void run() {
                if (PlaySoloActivity.this.mMultitudeFragment != null) {
                    PlaySoloActivity.this.mMultitudeFragment.onConnected(i);
                }
                PlaySoloActivity.this.plugDeviceUIListFragment();
                if (i > 0) {
                    PlaySoloActivity.this.title_txt_status.setVisibility(8);
                }
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.PlaySoloActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                if (PlaySoloActivity.this.isMultitudeFragmentVisible) {
                    PlaySoloActivity.this.toggleMultitudeFragment();
                }
            }
        });
        getWindow().addFlags(128);
        setContentView(R.layout.activity_play_solo);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        BleManager.getInstance(this).writeShutdown();
        this.mHandler.removeCallbacksAndMessages(null);
        ControlRuleManager.getInstance().removeHandleCallback();
        MyApplication.getApplication().setPlayLogs(this.productId, 1);
        unRegisterReceiver();
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(final int i) {
        LogUtil.e("plugDeviceUIListFragment onDisconnected size:" + i);
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlaySoloActivity.9
            @Override // java.lang.Runnable
            public void run() {
                if (PlaySoloActivity.this.mMultitudeFragment != null) {
                    PlaySoloActivity.this.mMultitudeFragment.onDisconnected(i);
                }
                PlaySoloActivity.this.plugDeviceUIListFragment();
                if (i == 0) {
                    PlaySoloActivity.this.title_txt_status.setVisibility(0);
                }
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.FragmentInteractionListener
    public void onFragmentRemoved(String str) {
        this.isMultitudeFragmentVisible = false;
    }

    public void setPlayLogs(int i) {
        String str = (String) SharedPreferencesUtil.get(MyApplication.getApplication(), Constans.Token);
        if (TextUtils.isEmpty(str) || TextUtils.isEmpty(this.productId)) {
            return;
        }
        this.baseRequest.getApi().setPlayLogs(str, this.productId, i).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, null, false) { // from class: com.hismith.hismithapp.PlaySoloActivity.6
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
            }
        });
    }

    public void shakeAnimation(View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(this, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }
}
