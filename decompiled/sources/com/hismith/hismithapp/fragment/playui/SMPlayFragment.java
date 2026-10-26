package com.hismith.hismithapp.fragment.playui;

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
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.ScaleAnimation;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.facebook.appevents.integrity.IntegrityManager;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.LazyLoadFragment;
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
public class SMPlayFragment extends LazyLoadFragment implements View.OnClickListener, SensorEventListener {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private ImageView img_reversal;
    private ConstraintLayout layout_action;
    private long leftExitTime;
    private IplayModeAdapter mAdapter;
    private String mAddress;
    private AnimatorSet mAnimatorSet;
    private BothWayArrowView mArrowBoth;
    private BleSpeedUtil mBleSpeedUtil;
    private ImageView mImgCloseMode;
    private ImageView mImgCloseVibration;
    private ImageView mImgGesture;
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
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_SHOW_HEART = 4;
    private final int WHAT_HIDE_HEART = 3;
    private final int WHAT_VIBRATION = 5;
    private final int WHAT_VIBRATION_MODE = 6;
    private final int WHAT_MUSIC = 7;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss:SSS");
    private boolean isPause = false;
    private boolean isReversal = false;
    private int mode = -1;
    private int vibrationMode = -1;
    private int music = -1;
    private boolean isMicMode = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        /* JADX WARN: Code restructure failed: missing block: B:45:0x01e7, code lost:
        
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
                    SMPlayFragment.this.mTxtSpeed.setText(intValue + "");
                    SMPlayFragment.this.mProgressSpeed.setProgress(intValue);
                    SMPlayFragment.this.isShowMode = 0;
                    if (SMPlayFragment.this.mLayoutMode.getVisibility() == 0) {
                        SMPlayFragment sMPlayFragment = SMPlayFragment.this;
                        sMPlayFragment.stopAnimation(sMPlayFragment.mLayoutMode);
                    }
                    SMPlayFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    SMPlayFragment.this.mode = 0;
                    if (SMPlayFragment.this.mAdapter != null) {
                        SMPlayFragment.this.mAdapter.setClickPosition(SMPlayFragment.this.mode - 1);
                        break;
                    }
                    break;
                case 2:
                    SMPlayFragment.this.mode = ((Integer) message.obj).intValue();
                    int i = SMPlayFragment.this.mode;
                    SMPlayFragment sMPlayFragment2 = SMPlayFragment.this;
                    if (i <= 0) {
                        sMPlayFragment2.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    } else {
                        sMPlayFragment2.isShowMode = 1;
                        if (SMPlayFragment.this.mLayoutMode.getVisibility() != 0) {
                            SMPlayFragment sMPlayFragment3 = SMPlayFragment.this;
                            sMPlayFragment3.startAnimation(sMPlayFragment3.mLayoutMode, SMPlayFragment.this.mLayoutVibration);
                        }
                        SMPlayFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                    }
                    if (SMPlayFragment.this.mAdapter != null) {
                        SMPlayFragment.this.mAdapter.setClickPosition(SMPlayFragment.this.mode - 1);
                        if (SMPlayFragment.this.mode != 0) {
                            SMPlayFragment.this.mRecyclerMode.smoothScrollToPosition(SMPlayFragment.this.mode);
                            break;
                        }
                    }
                    break;
                case 3:
                    SMPlayFragment.this.stopHeart();
                    break;
                case 4:
                    SMPlayFragment.this.startHeart();
                    break;
                case 5:
                    SMPlayFragment.this.isShowMode = 0;
                    if (SMPlayFragment.this.mLayoutVibration.getVisibility() == 0) {
                        SMPlayFragment sMPlayFragment4 = SMPlayFragment.this;
                        sMPlayFragment4.stopAnimation(sMPlayFragment4.mLayoutVibration);
                    }
                    SMPlayFragment.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    SMPlayFragment.this.vibrationMode = 0;
                    if (SMPlayFragment.this.mVibrationAdapter != null) {
                        SMPlayFragment.this.mVibrationAdapter.setClickPosition(SMPlayFragment.this.vibrationMode - 1);
                        break;
                    }
                    break;
                case 6:
                    SMPlayFragment.this.vibrationMode = ((Integer) message.obj).intValue();
                    int i2 = SMPlayFragment.this.vibrationMode;
                    SMPlayFragment sMPlayFragment5 = SMPlayFragment.this;
                    if (i2 >= 1) {
                        sMPlayFragment5.isShowMode = 2;
                        if (SMPlayFragment.this.mLayoutVibration.getVisibility() != 0) {
                            SMPlayFragment sMPlayFragment6 = SMPlayFragment.this;
                            sMPlayFragment6.startAnimation(sMPlayFragment6.mLayoutVibration, SMPlayFragment.this.mLayoutMode);
                        }
                        SMPlayFragment.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                    } else {
                        sMPlayFragment5.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    }
                    if (SMPlayFragment.this.mVibrationAdapter != null) {
                        SMPlayFragment.this.mVibrationAdapter.setClickPosition(SMPlayFragment.this.vibrationMode - 1);
                        if (SMPlayFragment.this.vibrationMode != 0) {
                            SMPlayFragment.this.mRecyclerVibration.smoothScrollToPosition(SMPlayFragment.this.vibrationMode);
                            break;
                        }
                    }
                    break;
                case 7:
                    int intValue2 = ((Integer) message.obj).intValue();
                    SMPlayFragment.this.music = intValue2;
                    SMPlayFragment sMPlayFragment7 = SMPlayFragment.this;
                    if (intValue2 <= 0) {
                        sMPlayFragment7.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice);
                        break;
                    } else {
                        sMPlayFragment7.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice_select);
                        break;
                    }
            }
        }
    });
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private long triggerTime = 0;
    private long triggerLastTime = 0;
    private int shakeCount = 0;
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.19
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                LogUtil.e("广播 RECEIVER_PRODUCT_INFO");
                SMPlayFragment.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                String stringExtra4 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                if (TextUtils.isEmpty(stringExtra4) || !stringExtra4.equals(MyApplication.getApplication().getDeviceAddress())) {
                    return;
                }
                if ("AA".equals(stringExtra)) {
                    SMPlayFragment.this.initBleAAData(stringExtra2, stringExtra3);
                } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    SMPlayFragment.this.initBleCCData(stringExtra2, stringExtra3);
                }
            }
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public void hideGestureTip() {
        if (this.mLayoutArrow.getVisibility() == 0) {
            this.mLayoutArrow.setVisibility(8);
            this.mArrowBoth.setOnValueListener(null);
        }
    }

    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:59:0x022e -> B:54:0x0231). Please report as a decompilation issue!!! */
    private void init() {
        PlayView_Four playView_Four;
        this.recorderVolumeManager = new RecorderVolumeManager();
        this.baseRequest = new RetrofitRequest();
        initSensor();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.mRecyclerMode.setLayoutManager(gridLayoutManager);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this.baseActivity, 2);
        this.mVibrationLayoutManager = gridLayoutManager2;
        this.mRecyclerVibration.setLayoutManager(gridLayoutManager2);
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = iplayModeAdapter;
        iplayModeAdapter.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.2
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) SMPlayFragment.this.mModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(SMPlayFragment.this.baseActivity).writeData(str);
            }
        });
        this.mRecyclerMode.setAdapter(this.mAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        iplayModeAdapter2.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.3
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) SMPlayFragment.this.mVibrationModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(SMPlayFragment.this.baseActivity).writeData(str);
            }
        });
        this.mRecyclerVibration.setAdapter(this.mVibrationAdapter);
        if (BleManager.getInstance(this.baseActivity).getmConnBleDevice() != null) {
            this.mTxtSpeed.setText(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed() + "");
            this.mProgressSpeed.setProgress(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed());
            if (MyApplication.getApplication().getmProductModel() != null && (playView_Four = this.mPlayView) != null) {
                playView_Four.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
            try {
                boolean hasMode = ProductUtil.hasMode(0);
                FrameLayout frameLayout = this.mLayoutModeBtn;
                if (hasMode) {
                    frameLayout.setVisibility(0);
                    this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
                    int mode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMode();
                    this.mode = mode;
                    if (mode > 0) {
                        this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                        IplayModeAdapter iplayModeAdapter3 = this.mAdapter;
                        if (iplayModeAdapter3 != null) {
                            iplayModeAdapter3.setClickPosition(this.mode - 1);
                        }
                    }
                } else {
                    frameLayout.setVisibility(8);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                boolean hasMode2 = ProductUtil.hasMode(1);
                FrameLayout frameLayout2 = this.mLayoutVibrateBtn;
                if (hasMode2) {
                    frameLayout2.setVisibility(0);
                    this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
                    int vibrationMode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibrationMode();
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
                    frameLayout2.setVisibility(8);
                }
            } catch (Exception e2) {
                e2.printStackTrace();
            }
            try {
                if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getMusic() != 1) {
                    this.mLayoutMusicBtn.setVisibility(8);
                } else {
                    this.mLayoutMusicBtn.setVisibility(0);
                    int music = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMusic();
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
            try {
                if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getReversal() != 1) {
                    this.img_reversal.clearAnimation();
                    this.img_reversal.setVisibility(8);
                } else {
                    this.img_reversal.clearAnimation();
                    this.img_reversal.setVisibility(0);
                    int reversal = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getReversal();
                    ImageView imageView3 = this.img_reversal;
                    if (reversal == 1) {
                        AnimUtil.rotate(imageView3, 1);
                        this.img_reversal.setRotationY(180.0f);
                    } else {
                        AnimUtil.rotate(imageView3, 0);
                        this.img_reversal.setRotationY(0.0f);
                    }
                }
            } catch (Exception e5) {
                e5.printStackTrace();
            }
        }
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
            case "01":
                try {
                    if ("A1".equals(str2) && MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                        this.mHandler.sendEmptyMessage(4);
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
                    Message message = new Message();
                    message.what = 1;
                    message.obj = 0;
                    this.mHandler.sendMessage(message);
                    Message message2 = new Message();
                    message2.what = 5;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    Message message3 = new Message();
                    message3.what = 7;
                    message3.obj = 0;
                    this.mHandler.sendMessage(message3);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
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
                } catch (Exception e3) {
                    e3.printStackTrace();
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
                } catch (Exception e4) {
                    e4.printStackTrace();
                    return;
                }
            case "06":
                try {
                    if (!"1003".equals(MyApplication.getApplication().getDeviceCode()) || !"00".equals(str2)) {
                        int intValue3 = Integer.valueOf(str2, 16).intValue();
                        Message message6 = new Message();
                        message6.what = 6;
                        message6.obj = Integer.valueOf(intValue3);
                        this.mHandler.sendMessage(message6);
                        break;
                    } else {
                        this.mHandler.sendEmptyMessage(4);
                        break;
                    }
                } catch (Exception e5) {
                    e5.printStackTrace();
                    return;
                }
                break;
            case "07":
                try {
                    if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getMusic() == 1) {
                        Message message7 = new Message();
                        message7.what = 7;
                        message7.obj = Integer.valueOf(str2, 16);
                        this.mHandler.sendMessage(message7);
                        break;
                    }
                } catch (Exception e6) {
                    e6.printStackTrace();
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
                            message.what = 7;
                            message.obj = 1;
                            this.mHandler.sendMessage(message);
                            break;
                        }
                    } else {
                        message.what = 7;
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
                    message4.what = 7;
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
            case "05":
                try {
                    int intValue3 = Integer.valueOf(str2, 16).intValue();
                    if (intValue3 > 100) {
                        i = 100;
                    } else if (intValue3 >= 0) {
                        i = intValue3;
                    }
                    Message message7 = new Message();
                    message7.what = 5;
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
                    message8.what = 6;
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
        this.mModes.clear();
        boolean hasMode = ProductUtil.hasMode(0);
        FrameLayout frameLayout = this.mLayoutModeBtn;
        if (hasMode) {
            frameLayout.setVisibility(0);
            this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
        } else {
            frameLayout.setVisibility(8);
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
        this.mVibrationModes.clear();
        boolean hasMode2 = ProductUtil.hasMode(1);
        FrameLayout frameLayout2 = this.mLayoutVibrateBtn;
        if (hasMode2) {
            frameLayout2.setVisibility(0);
            this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
        } else {
            frameLayout2.setVisibility(8);
        }
        this.mVibrationAdapter.notifyDataSetChanged();
        scrollVibrateHideImg();
        int music = productModel.getMusic();
        FrameLayout frameLayout3 = this.mLayoutMusicBtn;
        if (music == 1) {
            frameLayout3.setVisibility(0);
        } else {
            frameLayout3.setVisibility(8);
        }
        if (productModel.getShake() > 0) {
            this.mLayoutShakeBtn.setVisibility(0);
        } else {
            this.mBleSpeedUtil.setShake(false);
            this.mLayoutShakeBtn.setVisibility(8);
        }
        int reversal = productModel.getReversal();
        ImageView imageView = this.img_reversal;
        if (reversal != 1) {
            imageView.clearAnimation();
            this.img_reversal.setVisibility(8);
        } else {
            imageView.clearAnimation();
            this.img_reversal.setVisibility(0);
            AnimUtil.rotate(this.img_reversal, 0);
            this.img_reversal.setRotationY(0.0f);
        }
    }

    private void initRecordPermission() {
        if (ContextCompat.checkSelfPermission(this.baseActivity, "android.permission.RECORD_AUDIO") == 0) {
            startMicClick();
            return;
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                SMPlayFragment.this.m605xd9e2bd72((Permission) obj);
            }
        });
        if (this.baseActivity.mCompositeDisposable == null || this.baseActivity.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.baseActivity.mCompositeDisposable.add(subscribe);
    }

    private void initSensor() {
        SensorManager sensorManager = (SensorManager) this.baseActivity.getSystemService("sensor");
        this.mSensorManager = sensorManager;
        Sensor defaultSensor = sensorManager.getDefaultSensor(1);
        this.mSensor = defaultSensor;
        this.mSensorManager.registerListener(this, defaultSensor, 2);
        this.mBleSpeedUtil = new BleSpeedUtil(this.baseActivity);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
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
        this.mViewTouch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                SMPlayFragment.this.hideGestureTip();
                return false;
            }
        });
        this.mRecyclerMode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.5
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                SMPlayFragment.this.scrollModeHideImg();
            }
        });
        this.mRecyclerVibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.6
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                SMPlayFragment.this.scrollVibrateHideImg();
            }
        });
        this.mPlayView.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.7
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                LogUtil.e("onMoveChange speed:" + i);
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != SMPlayFragment.this.mProgressSpeed.getProgress() || i == 0) {
                    SMPlayFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (SMPlayFragment.this.mode != 0) {
                        SMPlayFragment.this.mode = 0;
                        if (SMPlayFragment.this.mAdapter != null) {
                            SMPlayFragment.this.mAdapter.setClickPosition(SMPlayFragment.this.mode - 1);
                        }
                    }
                    BleManager.getInstance(SMPlayFragment.this.baseActivity).writeSpeed(i);
                }
            }
        });
        this.mArrowBoth.setOnValueListener(new BothWayArrowView.OnValueListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.8
            @Override // com.hismith.hismithapp.view.BothWayArrowView.OnValueListener
            public void onValueChange(float f) {
                SMPlayFragment.this.mImgGesture.setRotation(f * (-45.0f));
            }
        });
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.9
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
                BleDeviceModel bleDeviceModel = BleManager.getInstance(SMPlayFragment.this.baseActivity).getmConnBleDevice();
                if (bleDeviceModel == null || i != bleDeviceModel.getSpeed()) {
                    BleManager.getInstance(SMPlayFragment.this.baseActivity).writeSpeed(i);
                }
            }
        });
        this.img_reversal.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.10
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    LogUtil.e("ACTION_DOWN");
                    AnimUtil.rotate(view, 1);
                    view.setRotationY(180.0f);
                    BleManager.getInstance(SMPlayFragment.this.baseActivity).writeOpenReversal();
                } else if (action == 1) {
                    LogUtil.e("ACTION_UP");
                    AnimUtil.rotate(view, 0);
                    view.setRotationY(0.0f);
                    BleManager.getInstance(SMPlayFragment.this.baseActivity).writeCloseReversal();
                }
                return true;
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.12
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                SMPlayFragment.this.inAnim = false;
                if (i != SMPlayFragment.this.isShowMode) {
                    SMPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                SMPlayFragment.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.13
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                SMPlayFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                SMPlayFragment.this.inAnim = true;
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
            BleManager.getInstance(this.baseActivity).writeSpeed(0);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.14
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                SMPlayFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != SMPlayFragment.this.isShowMode) {
                    SMPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                SMPlayFragment.this.inAnim = true;
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
            this.baseActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-fragment-playui-SMPlayFragment, reason: not valid java name */
    /* synthetic */ void m605xd9e2bd72(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            startMicClick();
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.11
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + SMPlayFragment.this.baseActivity.getPackageName()));
                    SMPlayFragment.this.startActivity(intent);
                    DialogUtil.dismissDialog();
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
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
            int i = this.music;
            BaseActivity baseActivity = this.baseActivity;
            if (i > 0) {
                BleManager.getInstance(baseActivity).writeCloseVoice();
                return;
            } else {
                BleManager.getInstance(baseActivity).writeOpenVoice();
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
            BleManager.getInstance(this.baseActivity).writeSpeed(this.mProgressSpeed.getProgress());
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
            BleManager.getInstance(this.baseActivity).writeCloseVibrate();
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

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_sm, (ViewGroup) null);
        this.mLayoutBg = (ImageView) inflate.findViewById(R.id.iplay_layout_bg);
        this.mPlayView = (PlayView_Four) inflate.findViewById(R.id.iplay_playView);
        this.mProgressSpeed = (ProgressBar) inflate.findViewById(R.id.iplay_progress_speed);
        this.mTxtMaxSpeed = (TextView) inflate.findViewById(R.id.iplay_txt_max_speed);
        this.mTxtMinSpeed = (TextView) inflate.findViewById(R.id.iplay_txt_min_speed);
        this.mTxtSpeed = (TextView) inflate.findViewById(R.id.iplay_txt_speed);
        this.mLayoutModeBtn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode_btn);
        this.mLayoutVibrateBtn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_btn);
        this.mLayoutMusicBtn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_music_btn);
        this.mLayoutMicBtn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mic_btn);
        this.mLayoutShakeBtn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_shake_btn);
        this.mImgMode = (ImageView) inflate.findViewById(R.id.iplay_img_mode);
        this.mImgVibrate = (ImageView) inflate.findViewById(R.id.iplay_img_vibrate);
        this.mImgMusic = (ImageView) inflate.findViewById(R.id.iplay_img_music);
        this.mImgMic = (ImageView) inflate.findViewById(R.id.iplay_img_mic);
        this.mImgShake = (ImageView) inflate.findViewById(R.id.iplay_img_shake);
        this.mLayoutMode = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode);
        this.mLayoutModeBorder = (LinearLayout) inflate.findViewById(R.id.iplay_layout_mode_border);
        this.mRecyclerMode = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_mode);
        this.mLayoutVibration = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibration);
        this.mLayoutVibrationBorder = (LinearLayout) inflate.findViewById(R.id.iplay_layout_vibration_border);
        this.mRecyclerVibration = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_vibration);
        this.mImgHeart = (ImageView) inflate.findViewById(R.id.iplay_img_heart);
        this.mImgCloseMode = (ImageView) inflate.findViewById(R.id.iplay_img_close_mode);
        this.mImgCloseVibration = (ImageView) inflate.findViewById(R.id.iplay_img_close_vibration);
        this.mLayoutArrowUp = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_up);
        this.mLayoutArrowDown = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_down);
        this.mLayoutVibrateArrowUp = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.mLayoutVibrateArrowDown = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.mViewTouch = inflate.findViewById(R.id.view_touch);
        this.mLayoutArrow = (LinearLayout) inflate.findViewById(R.id.layout_arrow);
        this.mArrowBoth = (BothWayArrowView) inflate.findViewById(R.id.arrow_both);
        this.mImgGesture = (ImageView) inflate.findViewById(R.id.img_gesture);
        this.mTxtSliding = (TextView) inflate.findViewById(R.id.txt_sliding);
        this.layout_action = (ConstraintLayout) inflate.findViewById(R.id.layout_action);
        this.img_reversal = (ImageView) inflate.findViewById(R.id.img_reversal);
        LanguageUtil.setText(this.baseActivity, this.mTxtSliding, "slidingControlSpeed");
        this.mPlayView.setProgressBar(this.mProgressSpeed);
        Glide.with(this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.mLayoutBg);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        unregisterReceiver();
        this.mHandler.removeCallbacksAndMessages(null);
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.cancelRecording();
        }
        ToastUtil.cancelToast();
    }

    @Override // androidx.fragment.app.Fragment
    public void onPause() {
        super.onPause();
        LogUtil.e("onPause()");
        if (this.mSensor != null) {
            this.mSensorManager.unregisterListener(this);
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
        LogUtil.e("onResume()");
        Sensor sensor = this.mSensor;
        if (sensor != null) {
            this.mSensorManager.registerListener(this, sensor, 2);
        }
    }

    @Override // android.hardware.SensorEventListener
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == 1) {
            float f = sensorEvent.values[0];
            float f2 = sensorEvent.values[1];
            float f3 = sensorEvent.values[2];
            long currentTimeMillis = System.currentTimeMillis() / 1000;
            long currentTimeMillis2 = System.currentTimeMillis() / 1000;
            this.mBleSpeedUtil.shakeSpeed(f, f2, f3);
            if (Math.abs(f) >= 20.0f || Math.abs(f2) >= 20.0f) {
                if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                    if (this.triggerTime == currentTimeMillis) {
                        int i = this.shakeCount + 1;
                        this.shakeCount = i;
                        if (i >= 3) {
                            LogUtil.e("Sensor value 摇晃");
                            this.recoverCount = 0;
                            this.isReversal = false;
                            boolean equals = "1003".equals(MyApplication.getApplication().getDeviceCode());
                            BaseActivity baseActivity = this.baseActivity;
                            if (equals) {
                                BleManager.getInstance(baseActivity).writeOpenClimax();
                            } else {
                                BleManager.getInstance(baseActivity).writeOpenClimax01();
                            }
                            startHeart();
                            this.triggerLastTime = 5 + currentTimeMillis;
                        }
                    } else {
                        this.shakeCount = 0;
                    }
                    this.triggerTime = currentTimeMillis;
                }
            } else if (f3 <= -9.0f && f3 > -10.0f) {
                int i2 = this.reversalCount + 1;
                this.reversalCount = i2;
                if (i2 >= 4) {
                    this.recoverCount = 0;
                    if (!this.isReversal) {
                        LogUtil.e("Sensor value 翻转翻转翻转");
                        this.isReversal = true;
                        BleManager.getInstance(this.baseActivity).writeSpeed(0);
                        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.15
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(SMPlayFragment.this.baseActivity).writeCloseVibrate();
                            }
                        }, 100L);
                        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.16
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(SMPlayFragment.this.baseActivity).writeCloseVoice();
                            }
                        }, 200L);
                    }
                }
            } else if (f3 > -1.0f) {
                int i3 = this.recoverCount + 1;
                this.recoverCount = i3;
                if (i3 >= 4) {
                    this.reversalCount = 0;
                    if (this.isReversal) {
                        LogUtil.e("Sensor value 翻转恢复");
                        this.isReversal = false;
                        if (this.mode <= 0) {
                            BleManager.getInstance(this.baseActivity).writeSpeed(this.mProgressSpeed.getProgress());
                        } else {
                            int size = this.mModes.size();
                            int i4 = this.mode;
                            BaseActivity baseActivity2 = this.baseActivity;
                            if (size >= i4) {
                                BleManager.getInstance(baseActivity2).writeData(this.mModes.get(this.mode - 1).getmCommand());
                            } else {
                                BleManager.getInstance(baseActivity2).writeSpeedMode(this.mode);
                            }
                        }
                        List<ProductModel.Mode> list = this.mVibrationModes;
                        if (list != null) {
                            int size2 = list.size();
                            int i5 = this.vibrationMode;
                            if (size2 >= i5 && i5 > 0) {
                                new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.17
                                    @Override // java.lang.Runnable
                                    public void run() {
                                        BleManager.getInstance(SMPlayFragment.this.baseActivity).writeData(((ProductModel.Mode) SMPlayFragment.this.mVibrationModes.get(SMPlayFragment.this.vibrationMode - 1)).getmCommand());
                                    }
                                }, 100L);
                            }
                        }
                        if (this.music > 0) {
                            new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.SMPlayFragment.18
                                @Override // java.lang.Runnable
                                public void run() {
                                    BleManager.getInstance(SMPlayFragment.this.baseActivity).writeOpenVoice();
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
    }
}
