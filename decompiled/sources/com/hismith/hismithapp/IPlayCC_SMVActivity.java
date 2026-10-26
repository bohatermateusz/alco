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
public class IPlayCC_SMVActivity extends BaseActivity implements View.OnClickListener, BleManager.OnConnectStateListener, SensorEventListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private static final float NS2S = 1.0E-9f;
    private RetrofitRequest baseRequest;
    private long leftExitTime;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private BothWayArrowView mArrowBoth;
    private BleSpeedUtil mBleSpeedUtil;
    private BleSpeedUtil mBleVibrateUtil;
    private ImageView mImgCloseLeft;
    private ImageView mImgCloseMode;
    private ImageView mImgCloseRight;
    private ImageView mImgCloseVibration;
    private ImageView mImgGesture;
    private ImageView mImgGestureLeft;
    private ImageView mImgGestureRight;
    private ImageView mImgHeart;
    private ImageView mImgMic;
    private ImageView mImgMicVibrate;
    private ImageView mImgMode;
    private ImageView mImgMusic;
    private ImageView mImgShake;
    private ImageView mImgShakeVibrate;
    private ImageView mImgVibrate;
    private LinearLayout mLayoutArrow;
    private FrameLayout mLayoutArrowDown;
    private FrameLayout mLayoutArrowUp;
    private ImageView mLayoutBg;
    private GridLayoutManager mLayoutManager;
    private FrameLayout mLayoutMicBtn;
    private FrameLayout mLayoutMicVibrateBtn;
    private FrameLayout mLayoutMode;
    private LinearLayout mLayoutModeBorder;
    private FrameLayout mLayoutModeBtn;
    private FrameLayout mLayoutMusicBtn;
    private FrameLayout mLayoutShakeBtn;
    private FrameLayout mLayoutShakeVibrateBtn;
    private FrameLayout mLayoutVibrateArrowDown;
    private FrameLayout mLayoutVibrateArrowUp;
    private FrameLayout mLayoutVibrateBtn;
    private FrameLayout mLayoutVibration;
    private LinearLayout mLayoutVibrationBorder;
    private List<ProductModel.Mode> mModes;
    private PlayView_Four mPlayView;
    private PlayView_Four mPlayViewVibrate;
    private ProgressBar mProgressSpeed;
    private ProgressBar mProgressVibrate;
    private RecyclerView mRecyclerMode;
    private RecyclerView mRecyclerVibration;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private TextView mTitleTxtName;
    private TextView mTitleTxtStatus;
    private TextView mTxtGestureTip;
    private TextView mTxtMaxSpeed;
    private TextView mTxtMaxVibrate;
    private TextView mTxtMinSpeed;
    private TextView mTxtMinVibrate;
    private TextView mTxtSliding;
    private TextView mTxtSpeed;
    private TextView mTxtVibrate;
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
    private final int WHAT_VIBRATE = 7;
    private int leftAction = 1;
    private int rightAction = 1;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss:SSS");
    private boolean isPause = false;
    private boolean isReversal = false;
    private int mode = -1;
    private int vibrationMode = -1;
    private int music = -1;
    private boolean isMicMode = false;
    private boolean isMicVibrateMode = false;
    private float[] angle = new float[3];
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        /* JADX WARN: Code restructure failed: missing block: B:45:0x0212, code lost:
        
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
                    IPlayCC_SMVActivity.this.mTxtSpeed.setText(intValue + "");
                    IPlayCC_SMVActivity.this.mProgressSpeed.setProgress(intValue);
                    IPlayCC_SMVActivity.this.isShowMode = 0;
                    if (IPlayCC_SMVActivity.this.mLayoutMode.getVisibility() == 0) {
                        IPlayCC_SMVActivity iPlayCC_SMVActivity = IPlayCC_SMVActivity.this;
                        iPlayCC_SMVActivity.stopAnimation(iPlayCC_SMVActivity.mLayoutMode);
                    }
                    IPlayCC_SMVActivity.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    IPlayCC_SMVActivity.this.mode = 0;
                    if (IPlayCC_SMVActivity.this.mAdapter != null) {
                        IPlayCC_SMVActivity.this.mAdapter.setClickPosition(IPlayCC_SMVActivity.this.mode - 1);
                        break;
                    }
                    break;
                case 2:
                    IPlayCC_SMVActivity.this.mode = ((Integer) message.obj).intValue();
                    int i = IPlayCC_SMVActivity.this.mode;
                    IPlayCC_SMVActivity iPlayCC_SMVActivity2 = IPlayCC_SMVActivity.this;
                    if (i <= 0) {
                        iPlayCC_SMVActivity2.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    } else {
                        iPlayCC_SMVActivity2.isShowMode = 1;
                        if (IPlayCC_SMVActivity.this.mLayoutMode.getVisibility() != 0) {
                            IPlayCC_SMVActivity iPlayCC_SMVActivity3 = IPlayCC_SMVActivity.this;
                            iPlayCC_SMVActivity3.startAnimation(iPlayCC_SMVActivity3.mLayoutMode, IPlayCC_SMVActivity.this.mLayoutVibration);
                        }
                        IPlayCC_SMVActivity.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                    }
                    if (IPlayCC_SMVActivity.this.mAdapter != null) {
                        IPlayCC_SMVActivity.this.mAdapter.setClickPosition(IPlayCC_SMVActivity.this.mode - 1);
                        if (IPlayCC_SMVActivity.this.mode != 0) {
                            IPlayCC_SMVActivity.this.mRecyclerMode.smoothScrollToPosition(IPlayCC_SMVActivity.this.mode);
                            break;
                        }
                    }
                    break;
                case 3:
                    IPlayCC_SMVActivity.this.stopHeart();
                    break;
                case 4:
                    IPlayCC_SMVActivity.this.startHeart();
                    break;
                case 5:
                    IPlayCC_SMVActivity.this.vibrationMode = ((Integer) message.obj).intValue();
                    int i2 = IPlayCC_SMVActivity.this.vibrationMode;
                    IPlayCC_SMVActivity iPlayCC_SMVActivity4 = IPlayCC_SMVActivity.this;
                    if (i2 >= 1) {
                        iPlayCC_SMVActivity4.isShowMode = 2;
                        if (IPlayCC_SMVActivity.this.mLayoutVibration.getVisibility() != 0) {
                            IPlayCC_SMVActivity iPlayCC_SMVActivity5 = IPlayCC_SMVActivity.this;
                            iPlayCC_SMVActivity5.startAnimation(iPlayCC_SMVActivity5.mLayoutVibration, IPlayCC_SMVActivity.this.mLayoutMode);
                        }
                        IPlayCC_SMVActivity.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                    } else {
                        iPlayCC_SMVActivity4.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    }
                    if (IPlayCC_SMVActivity.this.mVibrationAdapter != null) {
                        IPlayCC_SMVActivity.this.mVibrationAdapter.setClickPosition(IPlayCC_SMVActivity.this.vibrationMode - 1);
                        if (IPlayCC_SMVActivity.this.vibrationMode != 0) {
                            IPlayCC_SMVActivity.this.mRecyclerVibration.smoothScrollToPosition(IPlayCC_SMVActivity.this.vibrationMode);
                            break;
                        }
                    }
                    break;
                case 6:
                    int intValue2 = ((Integer) message.obj).intValue();
                    IPlayCC_SMVActivity.this.music = intValue2;
                    IPlayCC_SMVActivity iPlayCC_SMVActivity6 = IPlayCC_SMVActivity.this;
                    if (intValue2 <= 0) {
                        iPlayCC_SMVActivity6.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice);
                        break;
                    } else {
                        iPlayCC_SMVActivity6.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice_select);
                        break;
                    }
                case 7:
                    int intValue3 = ((Integer) message.obj).intValue();
                    IPlayCC_SMVActivity.this.mTxtVibrate.setText(intValue3 + "");
                    IPlayCC_SMVActivity.this.mProgressVibrate.setProgress(intValue3);
                    IPlayCC_SMVActivity.this.isShowMode = 0;
                    if (IPlayCC_SMVActivity.this.mLayoutVibration.getVisibility() == 0) {
                        IPlayCC_SMVActivity iPlayCC_SMVActivity7 = IPlayCC_SMVActivity.this;
                        iPlayCC_SMVActivity7.stopAnimation(iPlayCC_SMVActivity7.mLayoutVibration);
                    }
                    IPlayCC_SMVActivity.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    IPlayCC_SMVActivity.this.vibrationMode = 0;
                    if (IPlayCC_SMVActivity.this.mVibrationAdapter != null) {
                        IPlayCC_SMVActivity.this.mVibrationAdapter.setClickPosition(IPlayCC_SMVActivity.this.vibrationMode - 1);
                        break;
                    }
                    break;
            }
        }
    });
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.16
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayCC_SMVActivity.this.leftAction = 0;
                IPlayCC_SMVActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayCC_SMVActivity.this.lastLeftClickTime;
                IPlayCC_SMVActivity iPlayCC_SMVActivity = IPlayCC_SMVActivity.this;
                if (j < 500) {
                    iPlayCC_SMVActivity.finish();
                    return true;
                }
                iPlayCC_SMVActivity.lastLeftClickTime = currentTimeMillis;
                IPlayCC_SMVActivity.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.17
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayCC_SMVActivity.this.rightAction = 0;
                IPlayCC_SMVActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayCC_SMVActivity.this.lastRightClickTime;
                IPlayCC_SMVActivity iPlayCC_SMVActivity = IPlayCC_SMVActivity.this;
                if (j < 500) {
                    iPlayCC_SMVActivity.finish();
                    return true;
                }
                iPlayCC_SMVActivity.lastRightClickTime = currentTimeMillis;
                IPlayCC_SMVActivity.this.rightAction = 1;
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
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.25
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                IPlayCC_SMVActivity.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    IPlayCC_SMVActivity.this.initBleCCData(stringExtra2, stringExtra3);
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

    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:59:0x0184 -> B:30:0x0187). Please report as a decompilation issue!!! */
    private void init() {
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
        iplayModeAdapter.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.3
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) IPlayCC_SMVActivity.this.mModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(IPlayCC_SMVActivity.this).writeData(str);
            }
        });
        this.mRecyclerMode.setAdapter(this.mAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        iplayModeAdapter2.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.4
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) IPlayCC_SMVActivity.this.mVibrationModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(IPlayCC_SMVActivity.this).writeData(str);
            }
        });
        this.mRecyclerVibration.setAdapter(this.mVibrationAdapter);
        this.mTxtSpeed.setText(BleManager.getInstance(this).getmConnBleDevice().getSpeed() + "");
        this.mProgressSpeed.setProgress(BleManager.getInstance(this).getmConnBleDevice().getSpeed());
        this.mTxtVibrate.setText(BleManager.getInstance(this).getmConnBleDevice().getVibration() + "");
        this.mProgressVibrate.setProgress(BleManager.getInstance(this).getmConnBleDevice().getVibration());
        if (MyApplication.getApplication().getmProductModel() != null) {
            PlayView_Four playView_Four = this.mPlayView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
            PlayView_Four playView_Four2 = this.mPlayViewVibrate;
            if (playView_Four2 != null) {
                playView_Four2.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
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
                this.mBleVibrateUtil.setShake(false);
                this.mLayoutShakeBtn.setVisibility(8);
                this.mLayoutShakeVibrateBtn.setVisibility(8);
            } else {
                this.mLayoutShakeBtn.setVisibility(0);
                this.mLayoutShakeVibrateBtn.setVisibility(0);
            }
        } catch (Exception e4) {
            e4.printStackTrace();
        }
        BleManager.getInstance(this).initWriteMessage();
        AnimUtil.translate(this, this.mImgGestureLeft);
        AnimUtil.translate(this, this.mImgGestureRight);
        scrollModeHideImg();
        scrollVibrateHideImg();
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2) {
        int i;
        str.hashCode();
        i = 0;
        switch (str) {
            case "01":
                try {
                    if (("B0".equals(str2) || "B1".equals(str2)) && MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getMusic() == 1) {
                        Message message = new Message();
                        message.what = 6;
                        message.obj = Integer.valueOf(!"B0".equals(str2) ? 1 : 0);
                        this.mHandler.sendMessage(message);
                        break;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    return;
                }
                break;
            case "02":
                this.mHandler.sendEmptyMessage(3);
                try {
                    Message message2 = new Message();
                    message2.what = 1;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    Message message3 = new Message();
                    message3.what = 7;
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
                this.mHandler.sendEmptyMessage(3);
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
                this.mHandler.sendEmptyMessage(3);
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
            case "05":
                try {
                    int intValue3 = Integer.valueOf(str2, 16).intValue();
                    if (intValue3 > 100) {
                        i = 100;
                    } else if (intValue3 >= 0) {
                        i = intValue3;
                    }
                    Message message7 = new Message();
                    message7.what = 7;
                    message7.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message7);
                    break;
                } catch (Exception e5) {
                    e5.printStackTrace();
                    return;
                }
            case "06":
                try {
                    int intValue4 = Integer.valueOf(str2, 16).intValue();
                    Message message8 = new Message();
                    message8.what = 5;
                    message8.obj = Integer.valueOf(intValue4);
                    this.mHandler.sendMessage(message8);
                    break;
                } catch (Exception e6) {
                    e6.printStackTrace();
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
        PlayView_Four playView_Four2 = this.mPlayViewVibrate;
        if (playView_Four2 != null) {
            playView_Four2.setmScreenSpeed(productModel.getSlide());
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
            this.mLayoutShakeVibrateBtn.setVisibility(0);
        } else {
            this.mBleSpeedUtil.setShake(false);
            this.mBleVibrateUtil.setShake(false);
            this.mLayoutShakeBtn.setVisibility(8);
            this.mLayoutShakeVibrateBtn.setVisibility(8);
        }
    }

    private void initRecordPermission(final int i) {
        if (ContextCompat.checkSelfPermission(this, "android.permission.RECORD_AUDIO") == 0) {
            if (i == 1) {
                startMicClick();
                return;
            } else {
                startMicVibrateClick();
                return;
            }
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                IPlayCC_SMVActivity.this.m559x3fd0be7d(i, (Permission) obj);
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
        this.mBleVibrateUtil = new BleSpeedUtil(this);
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
        this.mPlayViewVibrate = (PlayView_Four) findViewById(R.id.iplay_playView_vibrate);
        this.mProgressVibrate = (ProgressBar) findViewById(R.id.iplay_progress_vibrate);
        this.mTxtMaxVibrate = (TextView) findViewById(R.id.iplay_txt_max_vibrate);
        this.mTxtMinVibrate = (TextView) findViewById(R.id.iplay_txt_min_vibrate);
        this.mTxtVibrate = (TextView) findViewById(R.id.iplay_txt_vibrate);
        this.mLayoutModeBtn = (FrameLayout) findViewById(R.id.iplay_layout_mode_btn);
        this.mLayoutVibrateBtn = (FrameLayout) findViewById(R.id.iplay_layout_vibrate_btn);
        this.mLayoutMusicBtn = (FrameLayout) findViewById(R.id.iplay_layout_music_btn);
        this.mLayoutMicBtn = (FrameLayout) findViewById(R.id.iplay_layout_mic_btn);
        this.mLayoutMicVibrateBtn = (FrameLayout) findViewById(R.id.iplay_layout_mic_vibrate_btn);
        this.mImgMode = (ImageView) findViewById(R.id.iplay_img_mode);
        this.mImgVibrate = (ImageView) findViewById(R.id.iplay_img_vibrate);
        this.mImgMusic = (ImageView) findViewById(R.id.iplay_img_music);
        this.mImgMic = (ImageView) findViewById(R.id.iplay_img_mic);
        this.mImgMicVibrate = (ImageView) findViewById(R.id.iplay_img_mic_vibrate);
        this.mLayoutShakeBtn = (FrameLayout) findViewById(R.id.iplay_layout_shake_btn);
        this.mLayoutShakeVibrateBtn = (FrameLayout) findViewById(R.id.iplay_layout_shake_vibrate_btn);
        this.mImgShake = (ImageView) findViewById(R.id.iplay_img_shake);
        this.mImgShakeVibrate = (ImageView) findViewById(R.id.iplay_img_shake_vibrate);
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
        this.mPlayViewVibrate.setProgressBar(this.mProgressVibrate);
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
        this.mLayoutMicVibrateBtn.setOnClickListener(this);
        this.mLayoutShakeBtn.setOnClickListener(this);
        this.mLayoutShakeVibrateBtn.setOnClickListener(this);
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
        this.mViewTouch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.5
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                IPlayCC_SMVActivity.this.hideGestureTip();
                return false;
            }
        });
        this.mRecyclerMode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.6
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                IPlayCC_SMVActivity.this.scrollModeHideImg();
            }
        });
        this.mRecyclerVibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.7
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                IPlayCC_SMVActivity.this.scrollVibrateHideImg();
            }
        });
        this.mPlayView.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.8
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != IPlayCC_SMVActivity.this.mProgressSpeed.getProgress() || i == 0) {
                    IPlayCC_SMVActivity.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (IPlayCC_SMVActivity.this.mode != 0) {
                        IPlayCC_SMVActivity.this.mode = 0;
                        if (IPlayCC_SMVActivity.this.mAdapter != null) {
                            IPlayCC_SMVActivity.this.mAdapter.setClickPosition(IPlayCC_SMVActivity.this.mode - 1);
                        }
                    }
                    BleManager.getInstance(IPlayCC_SMVActivity.this).writeSpeed(i);
                }
            }
        });
        this.mPlayViewVibrate.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.9
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != IPlayCC_SMVActivity.this.mProgressVibrate.getProgress() || i == 0) {
                    IPlayCC_SMVActivity.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    if (IPlayCC_SMVActivity.this.vibrationMode != 0) {
                        IPlayCC_SMVActivity.this.vibrationMode = 0;
                        if (IPlayCC_SMVActivity.this.mVibrationAdapter != null) {
                            IPlayCC_SMVActivity.this.mVibrationAdapter.setClickPosition(IPlayCC_SMVActivity.this.vibrationMode - 1);
                        }
                    }
                    BleManager.getInstance(IPlayCC_SMVActivity.this).writeVibrate(i);
                }
            }
        });
        this.mArrowBoth.setOnValueListener(new BothWayArrowView.OnValueListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.10
            @Override // com.hismith.hismithapp.view.BothWayArrowView.OnValueListener
            public void onValueChange(float f) {
                IPlayCC_SMVActivity.this.mImgGesture.setRotation(f * (-45.0f));
            }
        });
        BleManager.getInstance(this).setOnConnectStateListener(this);
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.11
            @Override // com.hismith.hismithapp.manager.RecorderVolumeManager.OnMicStatusListener
            public void onMicAmplitude(double d) {
                double d2 = d - 55.0d;
                if (d2 < 0.0d) {
                    d2 = 0.0d;
                }
                final int i = (int) (d2 * 2.0d);
                if (i > 100) {
                    i = 100;
                }
                if (IPlayCC_SMVActivity.this.isMicMode) {
                    BleManager.getInstance(IPlayCC_SMVActivity.this).writeSpeed(i == 0 ? 1 : i);
                }
                if (IPlayCC_SMVActivity.this.isMicVibrateMode) {
                    IPlayCC_SMVActivity.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.11.1
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(IPlayCC_SMVActivity.this).writeVibrate(i);
                        }
                    }, 50L);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.13
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayCC_SMVActivity.this.inAnim = false;
                if (i != IPlayCC_SMVActivity.this.isShowMode) {
                    IPlayCC_SMVActivity.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayCC_SMVActivity.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.14
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayCC_SMVActivity.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayCC_SMVActivity.this.inAnim = true;
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
        if (z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic);
            if (!this.isMicVibrateMode) {
                this.recorderVolumeManager.stopRecording();
            }
            BleManager.getInstance(this).writeSpeed(0);
            return;
        }
        imageView.setImageResource(R.mipmap.ic_iplay_mic_select);
        if (this.recorderVolumeManager.isRecording()) {
            return;
        }
        this.recorderVolumeManager.startRecording();
    }

    private void startMicVibrateClick() {
        boolean z = this.isMicVibrateMode;
        this.isMicVibrateMode = !z;
        ImageView imageView = this.mImgMicVibrate;
        if (z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic);
            if (!this.isMicMode) {
                this.recorderVolumeManager.stopRecording();
            }
            BleManager.getInstance(this).writeVibrate(0);
            return;
        }
        imageView.setImageResource(R.mipmap.ic_iplay_mic_select);
        if (this.recorderVolumeManager.isRecording()) {
            return;
        }
        this.recorderVolumeManager.startRecording();
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.15
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayCC_SMVActivity.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != IPlayCC_SMVActivity.this.isShowMode) {
                    IPlayCC_SMVActivity.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayCC_SMVActivity.this.inAnim = true;
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

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-IPlayCC_SMVActivity, reason: not valid java name */
    /* synthetic */ void m559x3fd0be7d(int i, Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            if (i == 1) {
                startMicClick();
                return;
            } else {
                startMicVibrateClick();
                return;
            }
        }
        if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            DialogUtil.showConfirmDialog(this, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.12
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + IPlayCC_SMVActivity.this.getPackageName()));
                    IPlayCC_SMVActivity.this.startActivity(intent);
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
            initRecordPermission(1);
            return;
        }
        if (id == R.id.iplay_layout_mic_vibrate_btn) {
            initRecordPermission(2);
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
        if (id == R.id.iplay_layout_shake_vibrate_btn) {
            boolean isShake2 = this.mBleVibrateUtil.isShake();
            BleSpeedUtil bleSpeedUtil2 = this.mBleVibrateUtil;
            if (isShake2) {
                bleSpeedUtil2.setShake(false);
                this.mImgShakeVibrate.setImageResource(R.mipmap.ic_iplay_shake);
                return;
            } else {
                bleSpeedUtil2.setShake(true);
                this.mImgShakeVibrate.setImageResource(R.mipmap.ic_iplay_shake_select);
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
            BleManager.getInstance(this).writeVibrate(this.mProgressVibrate.getProgress());
            this.vibrationMode = 0;
            this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
            IplayModeAdapter iplayModeAdapter2 = this.mVibrationAdapter;
            if (iplayModeAdapter2 != null) {
                iplayModeAdapter2.setClickPosition(this.vibrationMode - 1);
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
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.23
                @Override // java.lang.Runnable
                public void run() {
                    IPlayCC_SMVActivity.this.mTitleTxtStatus.setVisibility(8);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
            }
        });
        getWindow().addFlags(128);
        setContentView(R.layout.activity_iplay_cc_smv);
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
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.24
                @Override // java.lang.Runnable
                public void run() {
                    IPlayCC_SMVActivity.this.mTitleTxtStatus.setVisibility(0);
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
        this.mBleVibrateUtil.shakeSpeed(f2, f3, f4, new BleSpeedUtil.OnShakeListener() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.18
            @Override // com.hismith.hismithapp.utils.BleSpeedUtil.OnShakeListener
            public void onShake(int i) {
                BleDeviceModel bleDeviceModel = BleManager.getInstance(IPlayCC_SMVActivity.this).getmConnBleDevice();
                if (bleDeviceModel == null || i != bleDeviceModel.getVibration()) {
                    BleManager.getInstance(IPlayCC_SMVActivity.this).writeVibrate(i);
                }
            }
        });
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
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.19
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(IPlayCC_SMVActivity.this).writeVibrate(0);
                        }
                    }, 100L);
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.20
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(IPlayCC_SMVActivity.this).writeCloseVoice();
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
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.21
                        @Override // java.lang.Runnable
                        public void run() {
                            int i4 = IPlayCC_SMVActivity.this.vibrationMode;
                            IPlayCC_SMVActivity iPlayCC_SMVActivity = IPlayCC_SMVActivity.this;
                            if (i4 <= 0) {
                                BleManager.getInstance(iPlayCC_SMVActivity).writeVibrate(IPlayCC_SMVActivity.this.mProgressVibrate.getProgress());
                                return;
                            }
                            int size = iPlayCC_SMVActivity.mVibrationModes.size();
                            int i5 = IPlayCC_SMVActivity.this.vibrationMode;
                            IPlayCC_SMVActivity iPlayCC_SMVActivity2 = IPlayCC_SMVActivity.this;
                            if (size >= i5) {
                                BleManager.getInstance(iPlayCC_SMVActivity2).writeData(((ProductModel.Mode) IPlayCC_SMVActivity.this.mVibrationModes.get(IPlayCC_SMVActivity.this.vibrationMode - 1)).getmCommand());
                            } else {
                                BleManager.getInstance(iPlayCC_SMVActivity2).writeVibrateMode(IPlayCC_SMVActivity.this.vibrationMode);
                            }
                        }
                    }, 100L);
                    if (this.music > 0) {
                        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.IPlayCC_SMVActivity.22
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(IPlayCC_SMVActivity.this).writeOpenVoice();
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
