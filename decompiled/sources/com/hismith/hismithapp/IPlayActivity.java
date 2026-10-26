package com.hismith.hismithapp;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.view.animation.ScaleAnimation;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.BleSpeedUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class IPlayActivity extends BaseActivity implements View.OnClickListener, BleManager.OnConnectStateListener, SensorEventListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private static final float NS2S = 1.0E-9f;
    private RetrofitRequest baseRequest;
    private long leftExitTime;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private BothWayArrowView mArrowBoth;
    private BleSpeedUtil mBleSpeedUtil;
    private ImageView mImgCloseLeft;
    private ImageView mImgCloseMode;
    private ImageView mImgCloseRight;
    private ImageView mImgCloseVibration;
    private ImageView mImgGesture;
    private ImageView mImgGestureLeft;
    private ImageView mImgGestureRight;
    private ImageView mImgHeart;
    private ImageView mImgMic;
    private ImageView mImgMode;
    private ImageView mImgMusic;
    private ImageView mImgShake;
    private ImageView mImgVibrate;
    private LinearLayout mLayoutArrow;
    private FrameLayout mLayoutArrowDown;
    private FrameLayout mLayoutArrowUp;
    private ImageView mLayoutBg;
    private GridLayoutManager mLayoutManager;
    private FrameLayout mLayoutMicBtn;
    private FrameLayout mLayoutMode;
    private LinearLayout mLayoutModeBorder;
    private FrameLayout mLayoutModeBtn;
    private FrameLayout mLayoutMusicBtn;
    private FrameLayout mLayoutShakeBtn;
    private FrameLayout mLayoutVibrateArrowDown;
    private FrameLayout mLayoutVibrateArrowUp;
    private FrameLayout mLayoutVibrateBtn;
    private FrameLayout mLayoutVibration;
    private LinearLayout mLayoutVibrationBorder;
    private List<ProductModel.Mode> mModes;
    private PlayView_Four mPlayView;
    private ProgressBar mProgressSpeed;
    private RecyclerView mRecyclerMode;
    private RecyclerView mRecyclerVibration;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private TextView mTitleTxtName;
    private TextView mTitleTxtStatus;
    private TextView mTxtGestureTip;
    private TextView mTxtMaxSpeed;
    private TextView mTxtMinSpeed;
    private TextView mTxtSliding;
    private TextView mTxtSpeed;
    private IplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private List<ProductModel.Mode> mVibrationModes;
    private View mViewTouch;
    private RecorderVolumeManager recorderVolumeManager;
    private long rightExitTime;
    private float timestamp;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_SHOW_HEART = 4;
    private final int WHAT_HIDE_HEART = 3;
    private final int WHAT_VIBRATION_MODE = 5;
    private final int WHAT_MUSIC = 6;
    private int leftAction = 1;
    private int rightAction = 1;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss:SSS");
    private boolean isPause = false;
    private boolean isReversal = false;
    private int mode = -1;
    private int vibrationMode = -1;
    private int music = -1;
    private boolean isMicMode = false;
    private float[] angle = new float[3];
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.IPlayActivity.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        /* JADX WARN: Code restructure failed: missing block: B:39:0x01a3, code lost:
        
            return false;
         */
        @Override // android.os.Handler.Callback
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public boolean handleMessage(Message message) {
            switch (message.what) {
                case 1:
                    int intValue = ((Integer) message.obj).intValue();
                    IPlayActivity.this.mTxtSpeed.setText(intValue + "");
                    IPlayActivity.this.mProgressSpeed.setProgress(intValue);
                    IPlayActivity.this.isShowMode = 0;
                    if (IPlayActivity.this.mLayoutMode.getVisibility() == 0) {
                        IPlayActivity iPlayActivity = IPlayActivity.this;
                        iPlayActivity.stopAnimation(iPlayActivity.mLayoutMode);
                    }
                    IPlayActivity.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    IPlayActivity.this.mode = 0;
                    if (IPlayActivity.this.mAdapter != null) {
                        IPlayActivity.this.mAdapter.setClickPosition(IPlayActivity.this.mode - 1);
                        break;
                    }
                    break;
                case 2:
                    IPlayActivity.this.mode = ((Integer) message.obj).intValue();
                    int i = IPlayActivity.this.mode;
                    IPlayActivity iPlayActivity2 = IPlayActivity.this;
                    if (i <= 0) {
                        iPlayActivity2.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    } else {
                        iPlayActivity2.isShowMode = 1;
                        if (IPlayActivity.this.mLayoutMode.getVisibility() != 0) {
                            IPlayActivity iPlayActivity3 = IPlayActivity.this;
                            iPlayActivity3.startAnimation(iPlayActivity3.mLayoutMode, IPlayActivity.this.mLayoutVibration);
                        }
                        IPlayActivity.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                    }
                    if (IPlayActivity.this.mAdapter != null) {
                        IPlayActivity.this.mAdapter.setClickPosition(IPlayActivity.this.mode - 1);
                        if (IPlayActivity.this.mode != 0) {
                            IPlayActivity.this.mRecyclerMode.smoothScrollToPosition(IPlayActivity.this.mode);
                            break;
                        }
                    }
                    break;
                case 3:
                    IPlayActivity.this.stopHeart();
                    break;
                case 4:
                    IPlayActivity.this.startHeart();
                    break;
                case 5:
                    IPlayActivity.this.vibrationMode = ((Integer) message.obj).intValue();
                    int i2 = IPlayActivity.this.vibrationMode;
                    IPlayActivity iPlayActivity4 = IPlayActivity.this;
                    if (i2 >= 1) {
                        iPlayActivity4.isShowMode = 2;
                        if (IPlayActivity.this.mLayoutVibration.getVisibility() != 0) {
                            IPlayActivity iPlayActivity5 = IPlayActivity.this;
                            iPlayActivity5.startAnimation(iPlayActivity5.mLayoutVibration, IPlayActivity.this.mLayoutMode);
                        }
                        IPlayActivity.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                    } else {
                        iPlayActivity4.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    }
                    if (IPlayActivity.this.mVibrationAdapter != null) {
                        IPlayActivity.this.mVibrationAdapter.setClickPosition(IPlayActivity.this.vibrationMode - 1);
                        if (IPlayActivity.this.vibrationMode != 0) {
                            IPlayActivity.this.mRecyclerVibration.smoothScrollToPosition(IPlayActivity.this.vibrationMode);
                            break;
                        }
                    }
                    break;
                case 6:
                    int intValue2 = ((Integer) message.obj).intValue();
                    IPlayActivity.this.music = intValue2;
                    IPlayActivity iPlayActivity6 = IPlayActivity.this;
                    if (intValue2 <= 0) {
                        iPlayActivity6.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice);
                        break;
                    } else {
                        iPlayActivity6.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice_select);
                        break;
                    }
            }
        }
    });
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayActivity.15
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayActivity.this.leftAction = 0;
                IPlayActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayActivity.this.lastLeftClickTime;
                IPlayActivity iPlayActivity = IPlayActivity.this;
                if (j < 500) {
                    iPlayActivity.finish();
                    return true;
                }
                iPlayActivity.lastLeftClickTime = currentTimeMillis;
                IPlayActivity.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayActivity.16
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayActivity.this.rightAction = 0;
                IPlayActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayActivity.this.lastRightClickTime;
                IPlayActivity iPlayActivity = IPlayActivity.this;
                if (j < 500) {
                    iPlayActivity.finish();
                    return true;
                }
                iPlayActivity.lastRightClickTime = currentTimeMillis;
                IPlayActivity.this.rightAction = 1;
            }
            return true;
        }
    };
    private long triggerTime = 0;
    private long triggerLastTime = 0;
    private int shakeCount = 0;
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.IPlayActivity.23
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                IPlayActivity.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("AA".equals(stringExtra)) {
                    IPlayActivity.this.initBleAAData(stringExtra2, stringExtra3);
                } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    IPlayActivity.this.initBleCCData(stringExtra2, stringExtra3);
                }
            }
        }
    };

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
        if (this.mLayoutArrow.getVisibility() == 0) {
            this.mLayoutArrow.setVisibility(8);
            this.mArrowBoth.setOnValueListener(null);
        }
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

    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:48:0x01b9 -> B:44:0x01bc). Please report as a decompilation issue!!! */
    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:57:0x0147 -> B:29:0x014a). Please report as a decompilation issue!!! */
    private void init() {
        PlayView_Four playView_Four;
        this.recorderVolumeManager = new RecorderVolumeManager();
        this.baseRequest = new RetrofitRequest();
        initSensor();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 2);
        this.mLayoutManager = gridLayoutManager;
        this.mRecyclerMode.setLayoutManager(gridLayoutManager);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this, 2);
        this.mVibrationLayoutManager = gridLayoutManager2;
        this.mRecyclerVibration.setLayoutManager(gridLayoutManager2);
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this, this.mModes);
        this.mAdapter = iplayModeAdapter;
        iplayModeAdapter.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.IPlayActivity.3
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) IPlayActivity.this.mModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(IPlayActivity.this).writeData(str);
            }
        });
        this.mRecyclerMode.setAdapter(this.mAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        iplayModeAdapter2.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.IPlayActivity.4
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) IPlayActivity.this.mVibrationModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(IPlayActivity.this).writeData(str);
            }
        });
        this.mRecyclerVibration.setAdapter(this.mVibrationAdapter);
        if (BleManager.getInstance(this).getmConnBleDevice() != null) {
            this.mTxtSpeed.setText(BleManager.getInstance(this).getmConnBleDevice().getSpeed() + "");
            this.mProgressSpeed.setProgress(BleManager.getInstance(this).getmConnBleDevice().getSpeed());
            if (MyApplication.getApplication().getmProductModel() != null && (playView_Four = this.mPlayView) != null) {
                playView_Four.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
            try {
                if (ProductUtil.hasMode(0)) {
                    this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
                    int mode = BleManager.getInstance(this).getmConnBleDevice().getMode();
                    this.mode = mode;
                    if (mode > 0) {
                        this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                        IplayModeAdapter iplayModeAdapter3 = this.mAdapter;
                        if (iplayModeAdapter3 != null) {
                            iplayModeAdapter3.setClickPosition(this.mode - 1);
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                boolean hasMode = ProductUtil.hasMode(1);
                FrameLayout frameLayout = this.mLayoutVibrateBtn;
                if (hasMode) {
                    frameLayout.setVisibility(0);
                    this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
                    int vibrationMode = BleManager.getInstance(this).getmConnBleDevice().getVibrationMode();
                    this.vibrationMode = vibrationMode;
                    ImageView imageView = this.mImgVibrate;
                    if (vibrationMode >= 1) {
                        imageView.setImageResource(R.mipmap.ic_iplay_vibration_select);
                        IplayModeAdapter iplayModeAdapter4 = this.mVibrationAdapter;
                        if (iplayModeAdapter4 != null) {
                            iplayModeAdapter4.setClickPosition(this.vibrationMode - 1);
                        }
                    } else {
                        imageView.setImageResource(R.mipmap.ic_iplay_vibration);
                    }
                } else {
                    frameLayout.setVisibility(8);
                }
            } catch (Exception e2) {
                e2.printStackTrace();
            }
            try {
                if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getMusic() != 1) {
                    this.mLayoutMusicBtn.setVisibility(8);
                } else {
                    this.mLayoutMusicBtn.setVisibility(0);
                    int music = BleManager.getInstance(this).getmConnBleDevice().getMusic();
                    this.music = music;
                    ImageView imageView2 = this.mImgMusic;
                    if (music > 0) {
                        imageView2.setImageResource(R.mipmap.ic_iplay_voice_select);
                    } else {
                        imageView2.setImageResource(R.mipmap.ic_iplay_voice);
                    }
                }
            } catch (Exception e3) {
                e3.printStackTrace();
            }
            try {
                if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getShake() <= 0) {
                    this.mBleSpeedUtil.setShake(false);
                    this.mLayoutShakeBtn.setVisibility(8);
                } else {
                    this.mLayoutShakeBtn.setVisibility(0);
                }
            } catch (Exception e4) {
                e4.printStackTrace();
            }
        }
        BleManager.getInstance(this).initWriteMessage();
        AnimUtil.translate(this, this.mImgGestureLeft);
        AnimUtil.translate(this, this.mImgGestureRight);
        scrollModeHideImg();
        scrollVibrateHideImg();
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleAAData(String str, String str2) {
        int i;
        str.hashCode();
        i = 0;
        switch (str) {
            case "02":
                this.mHandler.sendEmptyMessage(3);
                try {
                    Message message = new Message();
                    message.what = 1;
                    message.obj = 0;
                    this.mHandler.sendMessage(message);
                    Message message2 = new Message();
                    message2.what = 5;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    Message message3 = new Message();
                    message3.what = 6;
                    message3.obj = 0;
                    this.mHandler.sendMessage(message3);
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                    return;
                }
            case "03":
            case "04":
                this.mHandler.sendEmptyMessage(3);
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message4 = new Message();
                    message4.what = 1;
                    message4.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message4);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "05":
                this.mHandler.sendEmptyMessage(3);
                try {
                    int intValue2 = Integer.valueOf(str2, 16).intValue();
                    Message message5 = new Message();
                    message5.what = 2;
                    message5.obj = Integer.valueOf(intValue2);
                    this.mHandler.sendMessage(message5);
                    break;
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
            case "06":
                try {
                    if (!"1003".equals(MyApplication.getApplication().getDeviceCode()) || !"00".equals(str2)) {
                        int intValue3 = Integer.valueOf(str2, 16).intValue();
                        Message message6 = new Message();
                        message6.what = 5;
                        message6.obj = Integer.valueOf(intValue3);
                        this.mHandler.sendMessage(message6);
                        break;
                    } else {
                        this.mHandler.sendEmptyMessage(4);
                        break;
                    }
                } catch (Exception e4) {
                    e4.printStackTrace();
                    return;
                }
                break;
            case "07":
                try {
                    if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getMusic() == 1) {
                        Message message7 = new Message();
                        message7.what = 6;
                        message7.obj = Integer.valueOf(str2, 16);
                        this.mHandler.sendMessage(message7);
                        break;
                    }
                } catch (Exception e5) {
                    e5.printStackTrace();
                    return;
                }
                break;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2) {
        int i;
        str.hashCode();
        i = 0;
        switch (str) {
            case "01":
                try {
                    Message message = new Message();
                    if (!"B0".equals(str2)) {
                        if ("B1".equals(str2)) {
                            message.what = 6;
                            message.obj = 1;
                            this.mHandler.sendMessage(message);
                            break;
                        }
                    } else {
                        message.what = 6;
                        message.obj = 0;
                        this.mHandler.sendMessage(message);
                        break;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;
            case "02":
                try {
                    Message message2 = new Message();
                    message2.what = 1;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    Message message3 = new Message();
                    message3.what = 5;
                    message3.obj = 0;
                    this.mHandler.sendMessage(message3);
                    Message message4 = new Message();
                    message4.what = 6;
                    message4.obj = 0;
                    this.mHandler.sendMessage(message4);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "03":
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message5 = new Message();
                    message5.what = 1;
                    message5.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message5);
                    break;
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
            case "04":
                try {
                    int intValue2 = Integer.valueOf(str2, 16).intValue();
                    Message message6 = new Message();
                    message6.what = 2;
                    message6.obj = Integer.valueOf(intValue2);
                    this.mHandler.sendMessage(message6);
                    break;
                } catch (Exception e4) {
                    e4.printStackTrace();
                    return;
                }
            case "06":
                try {
                    int intValue3 = Integer.valueOf(str2, 16).intValue();
                    Message message7 = new Message();
                    message7.what = 5;
                    message7.obj = Integer.valueOf(intValue3);
                    this.mHandler.sendMessage(message7);
                    break;
                } catch (Exception e5) {
                    e5.printStackTrace();
                    return;
                }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initModeShow() {
        int i = this.isShowMode;
        if (i != 0) {
            if (i == 1) {
                startAnimation(this.mLayoutMode, this.mLayoutVibration);
                return;
            } else {
                if (i == 2) {
                    startAnimation(this.mLayoutVibration, this.mLayoutMode);
                    return;
                }
                return;
            }
        }
        FrameLayout frameLayout = this.mLayoutMode;
        if (frameLayout != null && frameLayout.getVisibility() == 0) {
            stopAnimation(this.mLayoutMode);
        }
        FrameLayout frameLayout2 = this.mLayoutVibration;
        if (frameLayout2 == null || frameLayout2.getVisibility() != 0) {
            return;
        }
        stopAnimation(this.mLayoutVibration);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initProductData() {
        ProductModel productModel = MyApplication.getApplication().getmProductModel();
        LogUtil.e("BleManager 初始化：" + productModel);
        if (productModel == null) {
            return;
        }
        PlayView_Four playView_Four = this.mPlayView;
        if (playView_Four != null) {
            playView_Four.setmScreenSpeed(productModel.getSlide());
        }
        this.mModes.clear();
        if (ProductUtil.hasMode(0)) {
            this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
        this.mVibrationModes.clear();
        boolean hasMode = ProductUtil.hasMode(1);
        FrameLayout frameLayout = this.mLayoutVibrateBtn;
        if (hasMode) {
            frameLayout.setVisibility(0);
            this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
        } else {
            frameLayout.setVisibility(8);
        }
        this.mVibrationAdapter.notifyDataSetChanged();
        scrollVibrateHideImg();
        int music = productModel.getMusic();
        FrameLayout frameLayout2 = this.mLayoutMusicBtn;
        if (music == 1) {
            frameLayout2.setVisibility(0);
        } else {
            frameLayout2.setVisibility(8);
        }
        if (productModel.getShake() > 0) {
            this.mLayoutShakeBtn.setVisibility(0);
        } else {
            this.mBleSpeedUtil.setShake(false);
            this.mLayoutShakeBtn.setVisibility(8);
        }
    }

    private void initRecordPermission() {
        if (ContextCompat.checkSelfPermission(this, "android.permission.RECORD_AUDIO") == 0) {
            startMicClick();
            return;
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.IPlayActivity$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                IPlayActivity.this.m558xdd494b60((Permission) obj);
            }
        });
        if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    private void initSensor() {
        SensorManager sensorManager = (SensorManager) getSystemService("sensor");
        this.mSensorManager = sensorManager;
        this.mSensor = sensorManager.getDefaultSensor(1);
        this.mBleSpeedUtil = new BleSpeedUtil(this);
    }

    private void initView() {
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleTxtStatus = (TextView) findViewById(R.id.title_txt_status);
        this.mLayoutBg = (ImageView) findViewById(R.id.iplay_layout_bg);
        this.mPlayView = (PlayView_Four) findViewById(R.id.iplay_playView);
        this.mProgressSpeed = (ProgressBar) findViewById(R.id.iplay_progress_speed);
        this.mTxtMaxSpeed = (TextView) findViewById(R.id.iplay_txt_max_speed);
        this.mTxtMinSpeed = (TextView) findViewById(R.id.iplay_txt_min_speed);
        this.mTxtSpeed = (TextView) findViewById(R.id.iplay_txt_speed);
        this.mLayoutModeBtn = (FrameLayout) findViewById(R.id.iplay_layout_mode_btn);
        this.mLayoutVibrateBtn = (FrameLayout) findViewById(R.id.iplay_layout_vibrate_btn);
        this.mLayoutMusicBtn = (FrameLayout) findViewById(R.id.iplay_layout_music_btn);
        this.mLayoutMicBtn = (FrameLayout) findViewById(R.id.iplay_layout_mic_btn);
        this.mLayoutShakeBtn = (FrameLayout) findViewById(R.id.iplay_layout_shake_btn);
        this.mImgMode = (ImageView) findViewById(R.id.iplay_img_mode);
        this.mImgVibrate = (ImageView) findViewById(R.id.iplay_img_vibrate);
        this.mImgMusic = (ImageView) findViewById(R.id.iplay_img_music);
        this.mImgMic = (ImageView) findViewById(R.id.iplay_img_mic);
        this.mImgShake = (ImageView) findViewById(R.id.iplay_img_shake);
        this.mImgCloseLeft = (ImageView) findViewById(R.id.iplay_img_close_left);
        this.mImgCloseRight = (ImageView) findViewById(R.id.iplay_img_close_right);
        this.mLayoutMode = (FrameLayout) findViewById(R.id.iplay_layout_mode);
        this.mLayoutModeBorder = (LinearLayout) findViewById(R.id.iplay_layout_mode_border);
        this.mRecyclerMode = (RecyclerView) findViewById(R.id.iplay_recycler_mode);
        this.mLayoutVibration = (FrameLayout) findViewById(R.id.iplay_layout_vibration);
        this.mLayoutVibrationBorder = (LinearLayout) findViewById(R.id.iplay_layout_vibration_border);
        this.mRecyclerVibration = (RecyclerView) findViewById(R.id.iplay_recycler_vibration);
        this.mImgHeart = (ImageView) findViewById(R.id.iplay_img_heart);
        this.mImgCloseMode = (ImageView) findViewById(R.id.iplay_img_close_mode);
        this.mImgCloseVibration = (ImageView) findViewById(R.id.iplay_img_close_vibration);
        this.mLayoutArrowUp = (FrameLayout) findViewById(R.id.iplay_layout_arrow_up);
        this.mLayoutArrowDown = (FrameLayout) findViewById(R.id.iplay_layout_arrow_down);
        this.mLayoutVibrateArrowUp = (FrameLayout) findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.mLayoutVibrateArrowDown = (FrameLayout) findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.mViewTouch = findViewById(R.id.view_touch);
        this.mLayoutArrow = (LinearLayout) findViewById(R.id.layout_arrow);
        this.mArrowBoth = (BothWayArrowView) findViewById(R.id.arrow_both);
        this.mImgGesture = (ImageView) findViewById(R.id.img_gesture);
        this.mTxtSliding = (TextView) findViewById(R.id.txt_sliding);
        this.mImgGestureLeft = (ImageView) findViewById(R.id.iplay_img_gesture_left);
        this.mImgGestureRight = (ImageView) findViewById(R.id.iplay_img_gesture_right);
        this.mTxtGestureTip = (TextView) findViewById(R.id.iplay_txt_gesture_tip);
        LanguageUtil.setText(this, this.mTitleTxtName, "waitingRemoteConnection");
        LanguageUtil.setText(this, this.mTitleTxtStatus, "bluetoothDisconnected");
        LanguageUtil.setText(this, this.mTxtSliding, "slidingControlSpeed");
        LanguageUtil.setText(this, this.mTxtGestureTip, "exitDoubleGameNote");
        this.mPlayView.setProgressBar(this.mProgressSpeed);
        Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.mLayoutBg);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        if (this.mRecyclerMode.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.mLayoutArrowDown.getVisibility() == 4) {
                this.mLayoutArrowDown.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.mLayoutArrowDown.setVisibility(4);
        }
        if (!this.mRecyclerMode.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.mLayoutArrowUp.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.mLayoutArrowUp.getVisibility() == 4) {
                this.mLayoutArrowUp.setVisibility(0);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollVibrateHideImg() {
        if (this.mRecyclerVibration.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.mLayoutVibrateArrowDown.getVisibility() == 4) {
                this.mLayoutVibrateArrowDown.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.mLayoutVibrateArrowDown.setVisibility(4);
        }
        if (!this.mRecyclerVibration.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.mLayoutVibrateArrowUp.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.mLayoutVibrateArrowUp.getVisibility() == 4) {
                this.mLayoutVibrateArrowUp.setVisibility(0);
            }
        }
    }

    private void setListener() {
        this.mImgCloseLeft.setOnTouchListener(this.leftOnTouch);
        this.mImgCloseRight.setOnTouchListener(this.rightOnTouch);
        this.mLayoutModeBtn.setOnClickListener(this);
        this.mLayoutVibrateBtn.setOnClickListener(this);
        this.mLayoutMusicBtn.setOnClickListener(this);
        this.mLayoutMicBtn.setOnClickListener(this);
        this.mLayoutShakeBtn.setOnClickListener(this);
        this.mLayoutMode.setOnClickListener(this);
        this.mLayoutVibration.setOnClickListener(this);
        this.mLayoutModeBorder.setOnClickListener(this);
        this.mLayoutVibrationBorder.setOnClickListener(this);
        this.mImgCloseMode.setOnClickListener(this);
        this.mImgCloseVibration.setOnClickListener(this);
        this.mLayoutArrowUp.setOnClickListener(this);
        this.mLayoutArrowDown.setOnClickListener(this);
        this.mLayoutVibrateArrowUp.setOnClickListener(this);
        this.mLayoutVibrateArrowDown.setOnClickListener(this);
        this.mViewTouch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayActivity.5
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                IPlayActivity.this.hideGestureTip();
                return false;
            }
        });
        this.mRecyclerMode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.IPlayActivity.6
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                IPlayActivity.this.scrollModeHideImg();
            }
        });
        this.mRecyclerVibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.IPlayActivity.7
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                IPlayActivity.this.scrollVibrateHideImg();
            }
        });
        this.mPlayView.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.IPlayActivity.8
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                LogUtil.e("onMoveChange speed:" + i);
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != IPlayActivity.this.mProgressSpeed.getProgress() || i == 0) {
                    IPlayActivity.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (IPlayActivity.this.mode != 0) {
                        IPlayActivity.this.mode = 0;
                        if (IPlayActivity.this.mAdapter != null) {
                            IPlayActivity.this.mAdapter.setClickPosition(IPlayActivity.this.mode - 1);
                        }
                    }
                    BleManager.getInstance(IPlayActivity.this).writeSpeed(i);
                }
            }
        });
        this.mArrowBoth.setOnValueListener(new BothWayArrowView.OnValueListener() { // from class: com.hismith.hismithapp.IPlayActivity.9
            @Override // com.hismith.hismithapp.view.BothWayArrowView.OnValueListener
            public void onValueChange(float f) {
                IPlayActivity.this.mImgGesture.setRotation(f * (-45.0f));
            }
        });
        BleManager.getInstance(this).setOnConnectStateListener(this);
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.IPlayActivity.10
            @Override // com.hismith.hismithapp.manager.RecorderVolumeManager.OnMicStatusListener
            public void onMicAmplitude(double d) {
                double d2 = d - 55.0d;
                if (d2 < 0.0d) {
                    d2 = 0.0d;
                }
                int i = (int) (d2 * 2.0d);
                if (i > 100) {
                    i = 100;
                }
                BleDeviceModel bleDeviceModel = BleManager.getInstance(IPlayActivity.this).getmConnBleDevice();
                if (bleDeviceModel == null || i != bleDeviceModel.getSpeed()) {
                    BleManager.getInstance(IPlayActivity.this).writeSpeed(i);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void startAnimation(final FrameLayout frameLayout, final FrameLayout frameLayout2) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 0.0f, 1.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayActivity.12
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayActivity.this.inAnim = false;
                if (i != IPlayActivity.this.isShowMode) {
                    IPlayActivity.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayActivity.this.inAnim = true;
                frameLayout.setVisibility(0);
            }
        });
        if (frameLayout2 == null || frameLayout2.getVisibility() != 0) {
            if (frameLayout.getVisibility() != 0) {
                AnimatorSet animatorSet = new AnimatorSet();
                this.mAnimatorSet = animatorSet;
                animatorSet.playTogether(ofFloat, ofFloat2);
                this.mAnimatorSet.start();
                return;
            }
            return;
        }
        ObjectAnimator ofFloat3 = ObjectAnimator.ofFloat(frameLayout2, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat4 = ObjectAnimator.ofFloat(frameLayout2, "scaleY", 1.0f, 0.0f);
        ofFloat3.setDuration(this.animationDuration);
        ofFloat4.setDuration(this.animationDuration);
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayActivity.13
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayActivity.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayActivity.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet2 = new AnimatorSet();
        this.mAnimatorSet = animatorSet2;
        animatorSet2.play(ofFloat3).with(ofFloat4);
        this.mAnimatorSet.play(ofFloat3).before(ofFloat);
        this.mAnimatorSet.play(ofFloat).with(ofFloat2);
        this.mAnimatorSet.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void startHeart() {
        this.mImgHeart.setVisibility(0);
        ScaleAnimation scaleAnimation = new ScaleAnimation(1.0f, 0.7f, 1.0f, 0.7f, 1, 0.5f, 1, 0.5f);
        scaleAnimation.setDuration(250L);
        scaleAnimation.setFillAfter(true);
        scaleAnimation.setRepeatCount(-1);
        scaleAnimation.setRepeatMode(2);
        this.mImgHeart.startAnimation(scaleAnimation);
    }

    private void startMicClick() {
        boolean z = this.isMicMode;
        this.isMicMode = !z;
        ImageView imageView = this.mImgMic;
        if (!z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic_select);
            this.recorderVolumeManager.startRecording();
        } else {
            imageView.setImageResource(R.mipmap.ic_iplay_mic);
            this.recorderVolumeManager.stopRecording();
            BleManager.getInstance(this).writeSpeed(0);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void stopAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 1.0f, 0.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayActivity.14
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayActivity.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != IPlayActivity.this.isShowMode) {
                    IPlayActivity.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayActivity.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void stopHeart() {
        if (this.mImgHeart.getVisibility() == 0) {
            this.mImgHeart.clearAnimation();
            this.mImgHeart.setVisibility(8);
        }
    }

    private void unregisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-IPlayActivity, reason: not valid java name */
    /* synthetic */ void m558xdd494b60(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            startMicClick();
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            DialogUtil.showConfirmDialog(this, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.IPlayActivity.11
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + IPlayActivity.this.getPackageName()));
                    IPlayActivity.this.startActivity(intent);
                    DialogUtil.dismissDialog();
                }
            });
        }
    }

    @Override // android.hardware.SensorEventListener
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.iplay_layout_mode_btn) {
            if (this.inAnim) {
                return;
            }
            if (this.mLayoutMode.getVisibility() == 0) {
                stopAnimation(this.mLayoutMode);
                return;
            }
            this.isShowMode = 1;
            int visibility = this.mLayoutVibration.getVisibility();
            FrameLayout frameLayout = this.mLayoutMode;
            if (visibility == 0) {
                startAnimation(frameLayout, this.mLayoutVibration);
                return;
            } else {
                startAnimation(frameLayout, null);
                return;
            }
        }
        if (id == R.id.iplay_layout_vibrate_btn) {
            if (this.inAnim) {
                return;
            }
            if (this.mLayoutVibration.getVisibility() == 0) {
                stopAnimation(this.mLayoutVibration);
                return;
            }
            this.isShowMode = 2;
            int visibility2 = this.mLayoutMode.getVisibility();
            FrameLayout frameLayout2 = this.mLayoutVibration;
            if (visibility2 == 0) {
                startAnimation(frameLayout2, this.mLayoutMode);
                return;
            } else {
                startAnimation(frameLayout2, null);
                return;
            }
        }
        if (id == R.id.iplay_layout_music_btn) {
            if (this.music > 0) {
                BleManager.getInstance(this).writeCloseVoice();
                return;
            } else {
                BleManager.getInstance(this).writeOpenVoice();
                return;
            }
        }
        if (id == R.id.iplay_layout_mic_btn) {
            initRecordPermission();
            return;
        }
        if (id == R.id.iplay_layout_shake_btn) {
            boolean isShake = this.mBleSpeedUtil.isShake();
            BleSpeedUtil bleSpeedUtil = this.mBleSpeedUtil;
            if (isShake) {
                bleSpeedUtil.setShake(false);
                this.mImgShake.setImageResource(R.mipmap.ic_iplay_shake);
                return;
            } else {
                bleSpeedUtil.setShake(true);
                this.mImgShake.setImageResource(R.mipmap.ic_iplay_shake_select);
                return;
            }
        }
        if (id == R.id.iplay_layout_mode || id == R.id.iplay_layout_vibration) {
            if (this.mLayoutMode.getVisibility() == 0) {
                stopAnimation(this.mLayoutMode);
            }
            if (this.mLayoutVibration.getVisibility() == 0) {
                stopAnimation(this.mLayoutVibration);
                return;
            }
            return;
        }
        if (id == R.id.iplay_layout_mode_border || id == R.id.iplay_layout_vibration_border) {
            return;
        }
        if (id == R.id.iplay_img_close_mode) {
            BleManager.getInstance(this).writeSpeed(this.mProgressSpeed.getProgress());
            this.mode = 0;
            this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
            IplayModeAdapter iplayModeAdapter = this.mAdapter;
            if (iplayModeAdapter != null) {
                iplayModeAdapter.setClickPosition(this.mode - 1);
                return;
            }
            return;
        }
        if (id == R.id.iplay_img_close_vibration) {
            BleManager.getInstance(this).writeCloseVibrate();
            this.vibrationMode = 0;
            IplayModeAdapter iplayModeAdapter2 = this.mVibrationAdapter;
            if (iplayModeAdapter2 != null) {
                iplayModeAdapter2.setClickPosition(0 - 1);
                return;
            }
            return;
        }
        if (id == R.id.iplay_layout_arrow_up) {
            if (this.mRecyclerMode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.mRecyclerMode.smoothScrollToPosition(0);
            return;
        }
        if (id == R.id.iplay_layout_arrow_down) {
            if (this.mRecyclerMode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.mRecyclerMode.smoothScrollToPosition(this.mAdapter.getItemCount() - 1);
            return;
        }
        if (id == R.id.iplay_layout_vibrate_arrow_up) {
            if (this.mRecyclerVibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
                return;
            }
            this.mRecyclerVibration.smoothScrollToPosition(0);
            return;
        }
        if (id != R.id.iplay_layout_vibrate_arrow_down || this.mRecyclerVibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
            return;
        }
        this.mRecyclerVibration.smoothScrollToPosition(this.mVibrationAdapter.getItemCount() - 1);
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(int i) {
        if (i > 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayActivity.21
                @Override // java.lang.Runnable
                public void run() {
                    IPlayActivity.this.mTitleTxtStatus.setVisibility(8);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.IPlayActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
            }
        });
        getWindow().addFlags(128);
        setContentView(R.layout.activity_iplay);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        unregisterReceiver();
        this.mHandler.removeCallbacksAndMessages(null);
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.cancelRecording();
        }
        BleManager.getInstance(this).setOnConnectStateListener(null);
        BleManager.getInstance(this).writeShutdown();
        ToastUtil.cancelToast();
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(int i) {
        if (i == 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayActivity.22
                @Override // java.lang.Runnable
                public void run() {
                    IPlayActivity.this.mTitleTxtStatus.setVisibility(0);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
        if (this.mSensor != null) {
            this.mSensorManager.unregisterListener(this);
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
        Sensor sensor = this.mSensor;
        if (sensor != null) {
            this.mSensorManager.registerListener(this, sensor, 2);
        }
    }

    @Override // android.hardware.SensorEventListener
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == 4) {
            if (this.timestamp != 0.0f) {
                float f = (sensorEvent.timestamp - this.timestamp) * NS2S;
                float[] fArr = this.angle;
                fArr[0] = fArr[0] + (sensorEvent.values[0] * f);
                float[] fArr2 = this.angle;
                fArr2[1] = fArr2[1] + (sensorEvent.values[1] * f);
                float[] fArr3 = this.angle;
                fArr3[2] = fArr3[2] + (sensorEvent.values[2] * f);
                Math.toDegrees(this.angle[0]);
                Math.toDegrees(this.angle[1]);
                Math.toDegrees(this.angle[2]);
            }
            this.timestamp = sensorEvent.timestamp;
            return;
        }
        if (sensorEvent.sensor.getType() != 1) {
            sensorEvent.sensor.getType();
            return;
        }
        float f2 = sensorEvent.values[0];
        float f3 = sensorEvent.values[1];
        float f4 = sensorEvent.values[2];
        long currentTimeMillis = System.currentTimeMillis() / 1000;
        long currentTimeMillis2 = System.currentTimeMillis() / 1000;
        this.mBleSpeedUtil.shakeSpeed(f2, f3, f4);
        if (Math.abs(f2) >= 20.0f || Math.abs(f3) >= 20.0f) {
            if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                if (this.triggerTime == currentTimeMillis) {
                    int i = this.shakeCount + 1;
                    this.shakeCount = i;
                    if (i >= 3) {
                        LogUtil.e("Sensor value 摇晃");
                        this.recoverCount = 0;
                        this.isReversal = false;
                        if ("1003".equals(MyApplication.getApplication().getDeviceCode())) {
                            BleManager.getInstance(this).writeOpenClimax();
                        } else {
                            BleManager.getInstance(this).writeOpenClimax01();
                        }
                        startHeart();
                        this.triggerLastTime = 5 + currentTimeMillis;
                    }
                } else {
                    this.shakeCount = 0;
                }
                this.triggerTime = currentTimeMillis;
            }
        } else if (f4 <= -9.0f && f4 > -10.0f) {
            int i2 = this.reversalCount + 1;
            this.reversalCount = i2;
            if (i2 >= 4) {
                this.recoverCount = 0;
                if (!this.isReversal) {
                    LogUtil.e("Sensor value 翻转翻转翻转");
                    this.isReversal = true;
                    BleManager.getInstance(this).writeSpeed(0);
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayActivity.17
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(IPlayActivity.this).writeCloseVibrate();
                        }
                    }, 100L);
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayActivity.18
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(IPlayActivity.this).writeCloseVoice();
                        }
                    }, 200L);
                }
            }
        } else if (f4 > -1.0f) {
            int i3 = this.recoverCount + 1;
            this.recoverCount = i3;
            if (i3 >= 4) {
                this.reversalCount = 0;
                if (this.isReversal) {
                    LogUtil.e("Sensor value 翻转恢复");
                    this.isReversal = false;
                    if (this.mode <= 0) {
                        BleManager.getInstance(this).writeSpeed(this.mProgressSpeed.getProgress());
                    } else if (this.mModes.size() >= this.mode) {
                        BleManager.getInstance(this).writeData(this.mModes.get(this.mode - 1).getmCommand());
                    } else {
                        BleManager.getInstance(this).writeSpeedMode(this.mode);
                    }
                    if (this.vibrationMode > 0) {
                        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayActivity.19
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(IPlayActivity.this).writeData(((ProductModel.Mode) IPlayActivity.this.mVibrationModes.get(IPlayActivity.this.vibrationMode - 1)).getmCommand());
                            }
                        }, 100L);
                    }
                    if (this.music > 0) {
                        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayActivity.20
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(IPlayActivity.this).writeOpenVoice();
                            }
                        }, 200L);
                    }
                }
            }
        }
        if (this.reversalTime != currentTimeMillis2) {
            this.reversalTime = currentTimeMillis2;
            this.reversalCount = 0;
            this.recoverCount = 0;
        }
    }

    public void shakeAnimation(View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(this, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }
}
