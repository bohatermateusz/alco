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
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
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
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.BleSpeedUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class CC_SMVPlayFragment extends LazyLoadFragment implements View.OnClickListener, SensorEventListener {
    private BothWayArrowView arrow_both;
    private BaseActivity baseActivity;
    private ImageView img_gesture;
    private ImageView iplay_img_close_mode;
    private ImageView iplay_img_close_vibration;
    private ImageView iplay_img_mic;
    private ImageView iplay_img_mic_vibrate;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_shake;
    private ImageView iplay_img_shake_vibrate;
    private ImageView iplay_img_vibrate;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
    private ImageView iplay_layout_bg;
    private FrameLayout iplay_layout_mode;
    private LinearLayout iplay_layout_mode_border;
    private FrameLayout iplay_layout_vibrate_arrow_down;
    private FrameLayout iplay_layout_vibrate_arrow_up;
    private FrameLayout iplay_layout_vibration;
    private LinearLayout iplay_layout_vibration_border;
    private PlayView_Four iplay_playView;
    private PlayView_Four iplay_playView_vibrate;
    private VerticalProgressBar iplay_progress_speed;
    private VerticalProgressBar iplay_progress_vibrate;
    private RecyclerView iplay_recycler_mode;
    private RecyclerView iplay_recycler_vibration;
    private TextView iplay_txt_speed;
    private TextView iplay_txt_vibrate;
    private LinearLayout layout_arrow;
    private IplayModeAdapter mAdapter;
    private String mAddress;
    private AnimatorSet mAnimatorSet;
    private BleSpeedUtil mBleSpeedUtil;
    private BleSpeedUtil mBleVibrateUtil;
    private GridLayoutManager mLayoutManager;
    private List<ProductModel.Mode> mModes;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private IplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private List<ProductModel.Mode> mVibrationModes;
    private RecorderVolumeManager recorderVolumeManager;
    private TextView txt_sliding;
    private View view_touch;
    private int mode = -1;
    private int vibrationMode = -1;
    private boolean isMicMode = false;
    private boolean isMicVibrateMode = false;
    private boolean isReversal = false;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_VIBRATION_MODE = 5;
    private final int WHAT_VIBRATE = 7;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                int intValue = ((Integer) message.obj).intValue();
                CC_SMVPlayFragment.this.iplay_txt_speed.setText(intValue + "");
                CC_SMVPlayFragment.this.iplay_progress_speed.setProgress(intValue);
                CC_SMVPlayFragment.this.isShowMode = 0;
                if (CC_SMVPlayFragment.this.iplay_layout_mode.getVisibility() == 0) {
                    CC_SMVPlayFragment cC_SMVPlayFragment = CC_SMVPlayFragment.this;
                    cC_SMVPlayFragment.stopAnimation(cC_SMVPlayFragment.iplay_layout_mode);
                }
                if (CC_SMVPlayFragment.this.iplay_layout_vibration.getVisibility() == 0) {
                    CC_SMVPlayFragment cC_SMVPlayFragment2 = CC_SMVPlayFragment.this;
                    cC_SMVPlayFragment2.stopAnimation(cC_SMVPlayFragment2.iplay_layout_vibration);
                }
                CC_SMVPlayFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                CC_SMVPlayFragment.this.mode = 0;
                if (CC_SMVPlayFragment.this.mAdapter != null) {
                    CC_SMVPlayFragment.this.mAdapter.setClickPosition(CC_SMVPlayFragment.this.mode - 1);
                }
            } else if (i == 2) {
                CC_SMVPlayFragment.this.mode = ((Integer) message.obj).intValue();
                int i2 = CC_SMVPlayFragment.this.mode;
                CC_SMVPlayFragment cC_SMVPlayFragment3 = CC_SMVPlayFragment.this;
                if (i2 <= 0) {
                    cC_SMVPlayFragment3.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                } else {
                    cC_SMVPlayFragment3.isShowMode = 1;
                    if (CC_SMVPlayFragment.this.iplay_layout_mode.getVisibility() != 0) {
                        CC_SMVPlayFragment cC_SMVPlayFragment4 = CC_SMVPlayFragment.this;
                        cC_SMVPlayFragment4.startAnimation(cC_SMVPlayFragment4.iplay_layout_mode, CC_SMVPlayFragment.this.iplay_layout_vibration);
                    }
                    CC_SMVPlayFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                }
                if (CC_SMVPlayFragment.this.mAdapter != null) {
                    CC_SMVPlayFragment.this.mAdapter.setClickPosition(CC_SMVPlayFragment.this.mode - 1);
                    if (CC_SMVPlayFragment.this.mode > 0) {
                        CC_SMVPlayFragment.this.iplay_recycler_mode.smoothScrollToPosition(CC_SMVPlayFragment.this.mode - 1);
                    }
                }
            } else if (i == 5) {
                CC_SMVPlayFragment.this.vibrationMode = ((Integer) message.obj).intValue();
                int i3 = CC_SMVPlayFragment.this.vibrationMode;
                CC_SMVPlayFragment cC_SMVPlayFragment5 = CC_SMVPlayFragment.this;
                if (i3 >= 1) {
                    cC_SMVPlayFragment5.isShowMode = 2;
                    if (CC_SMVPlayFragment.this.iplay_layout_vibration.getVisibility() != 0) {
                        CC_SMVPlayFragment cC_SMVPlayFragment6 = CC_SMVPlayFragment.this;
                        cC_SMVPlayFragment6.startAnimation(cC_SMVPlayFragment6.iplay_layout_vibration, CC_SMVPlayFragment.this.iplay_layout_mode);
                    }
                    CC_SMVPlayFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                } else {
                    cC_SMVPlayFragment5.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                }
                if (CC_SMVPlayFragment.this.mVibrationAdapter != null) {
                    CC_SMVPlayFragment.this.mVibrationAdapter.setClickPosition(CC_SMVPlayFragment.this.vibrationMode - 1);
                    if (CC_SMVPlayFragment.this.vibrationMode > 0) {
                        CC_SMVPlayFragment.this.iplay_recycler_vibration.smoothScrollToPosition(CC_SMVPlayFragment.this.vibrationMode - 1);
                    }
                }
            } else if (i == 7) {
                int intValue2 = ((Integer) message.obj).intValue();
                CC_SMVPlayFragment.this.iplay_txt_vibrate.setText(intValue2 + "");
                CC_SMVPlayFragment.this.iplay_progress_vibrate.setProgress(intValue2);
                CC_SMVPlayFragment.this.isShowMode = 0;
                if (CC_SMVPlayFragment.this.iplay_layout_mode.getVisibility() == 0) {
                    CC_SMVPlayFragment cC_SMVPlayFragment7 = CC_SMVPlayFragment.this;
                    cC_SMVPlayFragment7.stopAnimation(cC_SMVPlayFragment7.iplay_layout_mode);
                }
                if (CC_SMVPlayFragment.this.iplay_layout_vibration.getVisibility() == 0) {
                    CC_SMVPlayFragment cC_SMVPlayFragment8 = CC_SMVPlayFragment.this;
                    cC_SMVPlayFragment8.stopAnimation(cC_SMVPlayFragment8.iplay_layout_vibration);
                }
                CC_SMVPlayFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                CC_SMVPlayFragment.this.vibrationMode = 0;
                if (CC_SMVPlayFragment.this.mVibrationAdapter != null) {
                    CC_SMVPlayFragment.this.mVibrationAdapter.setClickPosition(CC_SMVPlayFragment.this.vibrationMode - 1);
                }
            }
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.11
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                CC_SMVPlayFragment.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                String stringExtra4 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if (TextUtils.isEmpty(stringExtra4) || !stringExtra4.equals(MyApplication.getApplication().getDeviceAddress())) {
                    return;
                }
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    CC_SMVPlayFragment.this.initBleCCData(stringExtra2, stringExtra3);
                }
            }
        }
    };
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;

    public CC_SMVPlayFragment() {
    }

    public CC_SMVPlayFragment(String str) {
        this.mAddress = str;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideGestureTip() {
        if (this.layout_arrow.getVisibility() == 0) {
            this.layout_arrow.setVisibility(8);
            this.arrow_both.setOnValueListener(null);
        }
    }

    private void init() {
        this.recorderVolumeManager = new RecorderVolumeManager();
        initSensor();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.iplay_recycler_mode.setLayoutManager(gridLayoutManager);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this.baseActivity, 2);
        this.mVibrationLayoutManager = gridLayoutManager2;
        this.iplay_recycler_vibration.setLayoutManager(gridLayoutManager2);
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = iplayModeAdapter;
        iplayModeAdapter.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.2
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) CC_SMVPlayFragment.this.mModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeData(str);
            }
        });
        this.iplay_recycler_mode.setAdapter(this.mAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        iplayModeAdapter2.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.3
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) CC_SMVPlayFragment.this.mVibrationModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeData(str);
            }
        });
        this.iplay_recycler_vibration.setAdapter(this.mVibrationAdapter);
        this.iplay_txt_speed.setText(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed() + "");
        this.iplay_progress_speed.setProgress(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed());
        this.iplay_txt_vibrate.setText(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibration() + "");
        this.iplay_progress_vibrate.setProgress(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibration());
        if (MyApplication.getApplication().getmProductModel() != null) {
            PlayView_Four playView_Four = this.iplay_playView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
            PlayView_Four playView_Four2 = this.iplay_playView_vibrate;
            if (playView_Four2 != null) {
                playView_Four2.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
        }
        try {
            if (ProductUtil.hasMode(0)) {
                this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
                int mode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMode();
                this.mode = mode;
                if (mode > 0) {
                    this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
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
            if (ProductUtil.hasMode(1)) {
                this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
                int vibrationMode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibrationMode();
                this.vibrationMode = vibrationMode;
                ImageView imageView = this.iplay_img_vibrate;
                if (vibrationMode >= 1) {
                    imageView.setImageResource(R.mipmap.ic_iplay_vibration_select);
                    IplayModeAdapter iplayModeAdapter4 = this.mVibrationAdapter;
                    if (iplayModeAdapter4 != null) {
                        iplayModeAdapter4.setClickPosition(this.vibrationMode - 1);
                    }
                } else {
                    imageView.setImageResource(R.mipmap.ic_iplay_vibration);
                }
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        try {
            if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getShake() <= 0) {
                this.mBleSpeedUtil.setShake(false);
                this.mBleVibrateUtil.setShake(false);
                this.iplay_img_shake.setVisibility(8);
                this.iplay_img_shake_vibrate.setVisibility(8);
            } else {
                this.iplay_img_shake.setVisibility(0);
                this.iplay_img_shake_vibrate.setVisibility(0);
            }
        } catch (Exception e3) {
            e3.printStackTrace();
        }
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
            case "02":
                try {
                    Message message = new Message();
                    message.what = 1;
                    message.obj = 0;
                    this.mHandler.sendMessage(message);
                    Message message2 = new Message();
                    message2.what = 7;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            case "03":
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message3 = new Message();
                    message3.what = 1;
                    message3.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message3);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "04":
                try {
                    int intValue2 = Integer.valueOf(str2, 16).intValue();
                    Message message4 = new Message();
                    message4.what = 2;
                    message4.obj = Integer.valueOf(intValue2);
                    this.mHandler.sendMessage(message4);
                    break;
                } catch (Exception e3) {
                    e3.printStackTrace();
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
                    Message message5 = new Message();
                    message5.what = 7;
                    message5.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message5);
                    break;
                } catch (Exception e4) {
                    e4.printStackTrace();
                    return;
                }
            case "06":
                try {
                    int intValue4 = Integer.valueOf(str2, 16).intValue();
                    Message message6 = new Message();
                    message6.what = 5;
                    message6.obj = Integer.valueOf(intValue4);
                    this.mHandler.sendMessage(message6);
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
                startAnimation(this.iplay_layout_mode, this.iplay_layout_vibration);
                return;
            } else {
                if (i == 2) {
                    startAnimation(this.iplay_layout_vibration, this.iplay_layout_mode);
                    return;
                }
                return;
            }
        }
        FrameLayout frameLayout = this.iplay_layout_mode;
        if (frameLayout != null && frameLayout.getVisibility() == 0) {
            stopAnimation(this.iplay_layout_mode);
        }
        FrameLayout frameLayout2 = this.iplay_layout_vibration;
        if (frameLayout2 == null || frameLayout2.getVisibility() != 0) {
            return;
        }
        stopAnimation(this.iplay_layout_vibration);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initProductData() {
        ProductModel productModel = MyApplication.getApplication().getmProductModel();
        LogUtil.e("BleManager 初始化：" + productModel);
        if (productModel == null) {
            return;
        }
        PlayView_Four playView_Four = this.iplay_playView;
        if (playView_Four != null) {
            playView_Four.setmScreenSpeed(productModel.getSlide());
        }
        PlayView_Four playView_Four2 = this.iplay_playView_vibrate;
        if (playView_Four2 != null) {
            playView_Four2.setmScreenSpeed(productModel.getSlide());
        }
        this.mModes.clear();
        if (ProductUtil.hasMode(0)) {
            this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
        this.mVibrationModes.clear();
        if (ProductUtil.hasMode(1)) {
            this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
        }
        this.mVibrationAdapter.notifyDataSetChanged();
        scrollVibrateHideImg();
        if (productModel.getShake() > 0) {
            this.iplay_img_shake.setVisibility(0);
            this.iplay_img_shake_vibrate.setVisibility(0);
        } else {
            this.mBleSpeedUtil.setShake(false);
            this.mBleVibrateUtil.setShake(false);
            this.iplay_img_shake.setVisibility(8);
            this.iplay_img_shake_vibrate.setVisibility(8);
        }
    }

    private void initRecordPermission(final int i) {
        if (ContextCompat.checkSelfPermission(this.baseActivity, "android.permission.RECORD_AUDIO") == 0) {
            if (i == 1) {
                startMicClick();
                return;
            } else {
                startMicVibrateClick();
                return;
            }
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                CC_SMVPlayFragment.this.m604x48b70575(i, (Permission) obj);
            }
        });
        if (this.baseActivity.mCompositeDisposable == null || this.baseActivity.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.baseActivity.mCompositeDisposable.add(subscribe);
    }

    private void initSensor() {
        LogUtil.e("initSensor()");
        SensorManager sensorManager = (SensorManager) this.baseActivity.getSystemService("sensor");
        this.mSensorManager = sensorManager;
        Sensor defaultSensor = sensorManager.getDefaultSensor(1);
        this.mSensor = defaultSensor;
        this.mSensorManager.registerListener(this, defaultSensor, 2);
        this.mBleSpeedUtil = new BleSpeedUtil(this.baseActivity);
        this.mBleVibrateUtil = new BleSpeedUtil(this.baseActivity);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        if (this.iplay_recycler_mode.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.iplay_layout_arrow_down.getVisibility() == 4) {
                this.iplay_layout_arrow_down.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.iplay_layout_arrow_down.setVisibility(4);
        }
        if (!this.iplay_recycler_mode.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.iplay_layout_arrow_up.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.iplay_layout_arrow_up.getVisibility() == 4) {
                this.iplay_layout_arrow_up.setVisibility(0);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollVibrateHideImg() {
        if (this.iplay_recycler_vibration.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.iplay_layout_vibrate_arrow_down.getVisibility() == 4) {
                this.iplay_layout_vibrate_arrow_down.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.iplay_layout_vibrate_arrow_down.setVisibility(4);
        }
        if (!this.iplay_recycler_vibration.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.iplay_layout_vibrate_arrow_up.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.iplay_layout_vibrate_arrow_up.getVisibility() == 4) {
                this.iplay_layout_vibrate_arrow_up.setVisibility(0);
            }
        }
    }

    private void setListener() {
        this.iplay_img_mode.setOnClickListener(this);
        this.iplay_img_vibrate.setOnClickListener(this);
        this.iplay_img_mic.setOnClickListener(this);
        this.iplay_img_mic_vibrate.setOnClickListener(this);
        this.iplay_img_shake.setOnClickListener(this);
        this.iplay_img_shake_vibrate.setOnClickListener(this);
        this.iplay_layout_mode.setOnClickListener(this);
        this.iplay_layout_vibration.setOnClickListener(this);
        this.iplay_layout_mode_border.setOnClickListener(this);
        this.iplay_layout_vibration_border.setOnClickListener(this);
        this.iplay_img_close_mode.setOnClickListener(this);
        this.iplay_img_close_vibration.setOnClickListener(this);
        this.iplay_layout_arrow_up.setOnClickListener(this);
        this.iplay_layout_arrow_down.setOnClickListener(this);
        this.iplay_layout_vibrate_arrow_up.setOnClickListener(this);
        this.iplay_layout_vibrate_arrow_down.setOnClickListener(this);
        this.view_touch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                CC_SMVPlayFragment.this.hideGestureTip();
                return false;
            }
        });
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.5
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                CC_SMVPlayFragment.this.scrollModeHideImg();
            }
        });
        this.iplay_recycler_vibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.6
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                CC_SMVPlayFragment.this.scrollVibrateHideImg();
            }
        });
        this.iplay_playView.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.7
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                LogUtil.e("onMoveChange speed:" + i);
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != CC_SMVPlayFragment.this.iplay_progress_speed.getProgress() || i == 0) {
                    CC_SMVPlayFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (CC_SMVPlayFragment.this.mode != 0) {
                        CC_SMVPlayFragment.this.mode = 0;
                        if (CC_SMVPlayFragment.this.mAdapter != null) {
                            CC_SMVPlayFragment.this.mAdapter.setClickPosition(CC_SMVPlayFragment.this.mode - 1);
                        }
                    }
                    BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeSpeed(i);
                }
            }
        });
        this.iplay_playView_vibrate.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.8
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                LogUtil.e("onMoveChange vibrate:" + i);
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != CC_SMVPlayFragment.this.iplay_progress_vibrate.getProgress() || i == 0) {
                    CC_SMVPlayFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    if (CC_SMVPlayFragment.this.vibrationMode != 0) {
                        CC_SMVPlayFragment.this.vibrationMode = 0;
                        if (CC_SMVPlayFragment.this.mVibrationAdapter != null) {
                            CC_SMVPlayFragment.this.mVibrationAdapter.setClickPosition(CC_SMVPlayFragment.this.vibrationMode - 1);
                        }
                    }
                    BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeVibrate(i);
                }
            }
        });
        this.arrow_both.setOnValueListener(new BothWayArrowView.OnValueListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.9
            @Override // com.hismith.hismithapp.view.BothWayArrowView.OnValueListener
            public void onValueChange(float f) {
                CC_SMVPlayFragment.this.img_gesture.setRotation(f * (-45.0f));
            }
        });
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.10
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
                if (CC_SMVPlayFragment.this.isMicMode) {
                    BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeSpeed(i);
                }
                if (CC_SMVPlayFragment.this.isMicVibrateMode) {
                    CC_SMVPlayFragment.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.10.1
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeVibrate(i);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.12
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                CC_SMVPlayFragment.this.inAnim = false;
                if (i != CC_SMVPlayFragment.this.isShowMode) {
                    CC_SMVPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                CC_SMVPlayFragment.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.13
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                CC_SMVPlayFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                CC_SMVPlayFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet2 = new AnimatorSet();
        this.mAnimatorSet = animatorSet2;
        animatorSet2.play(ofFloat3).with(ofFloat4);
        this.mAnimatorSet.play(ofFloat3).before(ofFloat);
        this.mAnimatorSet.play(ofFloat).with(ofFloat2);
        this.mAnimatorSet.start();
    }

    private void startMicClick() {
        boolean z = this.isMicMode;
        this.isMicMode = !z;
        ImageView imageView = this.iplay_img_mic;
        if (z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic);
            if (!this.isMicVibrateMode) {
                this.recorderVolumeManager.stopRecording();
            }
            BleManager.getInstance(this.baseActivity).writeSpeed(0);
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
        ImageView imageView = this.iplay_img_mic_vibrate;
        if (z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic);
            if (!this.isMicMode) {
                this.recorderVolumeManager.stopRecording();
            }
            BleManager.getInstance(this.baseActivity).writeVibrate(0);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.14
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                CC_SMVPlayFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != CC_SMVPlayFragment.this.isShowMode) {
                    CC_SMVPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                CC_SMVPlayFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    private void unregisterReceiver() {
        try {
            this.baseActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-fragment-playui-CC_SMVPlayFragment, reason: not valid java name */
    /* synthetic */ void m604x48b70575(int i, Permission permission) throws Exception {
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
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.15
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + CC_SMVPlayFragment.this.baseActivity.getPackageName()));
                    CC_SMVPlayFragment.this.startActivity(intent);
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
        if (id == R.id.iplay_img_mode) {
            if (this.inAnim) {
                return;
            }
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
                return;
            }
            this.isShowMode = 1;
            int visibility = this.iplay_layout_vibration.getVisibility();
            FrameLayout frameLayout = this.iplay_layout_mode;
            if (visibility == 0) {
                startAnimation(frameLayout, this.iplay_layout_vibration);
                return;
            } else {
                startAnimation(frameLayout, null);
                return;
            }
        }
        if (id == R.id.iplay_img_vibrate) {
            if (this.inAnim) {
                return;
            }
            if (this.iplay_layout_vibration.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_vibration);
                return;
            }
            this.isShowMode = 2;
            int visibility2 = this.iplay_layout_mode.getVisibility();
            FrameLayout frameLayout2 = this.iplay_layout_vibration;
            if (visibility2 == 0) {
                startAnimation(frameLayout2, this.iplay_layout_mode);
                return;
            } else {
                startAnimation(frameLayout2, null);
                return;
            }
        }
        if (id == R.id.iplay_img_mic) {
            initRecordPermission(1);
            return;
        }
        if (id == R.id.iplay_img_mic_vibrate) {
            initRecordPermission(2);
            return;
        }
        if (id == R.id.iplay_img_shake) {
            boolean isShake = this.mBleSpeedUtil.isShake();
            BleSpeedUtil bleSpeedUtil = this.mBleSpeedUtil;
            if (isShake) {
                bleSpeedUtil.setShake(false);
                this.iplay_img_shake.setImageResource(R.mipmap.ic_iplay_shake);
                return;
            } else {
                bleSpeedUtil.setShake(true);
                this.iplay_img_shake.setImageResource(R.mipmap.ic_iplay_shake_select);
                return;
            }
        }
        if (id == R.id.iplay_img_shake_vibrate) {
            boolean isShake2 = this.mBleVibrateUtil.isShake();
            BleSpeedUtil bleSpeedUtil2 = this.mBleVibrateUtil;
            if (isShake2) {
                bleSpeedUtil2.setShake(false);
                this.iplay_img_shake_vibrate.setImageResource(R.mipmap.ic_iplay_shake);
                return;
            } else {
                bleSpeedUtil2.setShake(true);
                this.iplay_img_shake_vibrate.setImageResource(R.mipmap.ic_iplay_shake_select);
                return;
            }
        }
        if (id == R.id.iplay_layout_mode || id == R.id.iplay_layout_vibration) {
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
            }
            if (this.iplay_layout_vibration.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_vibration);
                return;
            }
            return;
        }
        if (id == R.id.iplay_layout_mode_border || id == R.id.iplay_layout_vibration_border) {
            return;
        }
        if (id == R.id.iplay_img_close_mode) {
            BleManager.getInstance(this.baseActivity).writeSpeed(this.iplay_progress_speed.getProgress());
            this.mode = 0;
            this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
            IplayModeAdapter iplayModeAdapter = this.mAdapter;
            if (iplayModeAdapter != null) {
                iplayModeAdapter.setClickPosition(this.mode - 1);
                return;
            }
            return;
        }
        if (id == R.id.iplay_img_close_vibration) {
            BleManager.getInstance(this.baseActivity).writeVibrate(this.iplay_progress_vibrate.getProgress());
            this.vibrationMode = 0;
            this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
            IplayModeAdapter iplayModeAdapter2 = this.mVibrationAdapter;
            if (iplayModeAdapter2 != null) {
                iplayModeAdapter2.setClickPosition(this.vibrationMode - 1);
                return;
            }
            return;
        }
        if (id == R.id.iplay_layout_arrow_up) {
            if (this.iplay_recycler_mode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.iplay_recycler_mode.smoothScrollToPosition(0);
            return;
        }
        if (id == R.id.iplay_layout_arrow_down) {
            if (this.iplay_recycler_mode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.iplay_recycler_mode.smoothScrollToPosition(this.mAdapter.getItemCount() - 1);
            return;
        }
        if (id == R.id.iplay_layout_vibrate_arrow_up) {
            if (this.iplay_recycler_vibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
                return;
            }
            this.iplay_recycler_vibration.smoothScrollToPosition(0);
            return;
        }
        if (id != R.id.iplay_layout_vibrate_arrow_down || this.iplay_recycler_vibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
            return;
        }
        this.iplay_recycler_vibration.smoothScrollToPosition(this.mVibrationAdapter.getItemCount() - 1);
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_cc_smv, (ViewGroup) null);
        this.iplay_layout_bg = (ImageView) inflate.findViewById(R.id.iplay_layout_bg);
        this.iplay_playView = (PlayView_Four) inflate.findViewById(R.id.iplay_playView);
        this.iplay_playView_vibrate = (PlayView_Four) inflate.findViewById(R.id.iplay_playView_vibrate);
        this.iplay_progress_speed = (VerticalProgressBar) inflate.findViewById(R.id.iplay_progress_speed);
        this.iplay_progress_vibrate = (VerticalProgressBar) inflate.findViewById(R.id.iplay_progress_vibrate);
        this.iplay_txt_speed = (TextView) inflate.findViewById(R.id.iplay_txt_speed);
        this.iplay_txt_vibrate = (TextView) inflate.findViewById(R.id.iplay_txt_vibrate);
        this.iplay_img_mode = (ImageView) inflate.findViewById(R.id.iplay_img_mode);
        this.iplay_img_vibrate = (ImageView) inflate.findViewById(R.id.iplay_img_vibrate);
        this.iplay_img_mic = (ImageView) inflate.findViewById(R.id.iplay_img_mic);
        this.iplay_img_mic_vibrate = (ImageView) inflate.findViewById(R.id.iplay_img_mic_vibrate);
        this.iplay_img_shake = (ImageView) inflate.findViewById(R.id.iplay_img_shake);
        this.iplay_img_shake_vibrate = (ImageView) inflate.findViewById(R.id.iplay_img_shake_vibrate);
        this.iplay_layout_mode = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode);
        this.iplay_layout_vibration = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibration);
        this.iplay_layout_mode_border = (LinearLayout) inflate.findViewById(R.id.iplay_layout_mode_border);
        this.iplay_layout_vibration_border = (LinearLayout) inflate.findViewById(R.id.iplay_layout_vibration_border);
        this.iplay_recycler_mode = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_mode);
        this.iplay_recycler_vibration = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_vibration);
        this.iplay_layout_arrow_up = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_down);
        this.iplay_layout_vibrate_arrow_up = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.iplay_layout_vibrate_arrow_down = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.iplay_img_close_mode = (ImageView) inflate.findViewById(R.id.iplay_img_close_mode);
        this.iplay_img_close_vibration = (ImageView) inflate.findViewById(R.id.iplay_img_close_vibration);
        this.layout_arrow = (LinearLayout) inflate.findViewById(R.id.layout_arrow);
        this.arrow_both = (BothWayArrowView) inflate.findViewById(R.id.arrow_both);
        this.img_gesture = (ImageView) inflate.findViewById(R.id.img_gesture);
        this.txt_sliding = (TextView) inflate.findViewById(R.id.txt_sliding);
        this.view_touch = inflate.findViewById(R.id.view_touch);
        LanguageUtil.setText(this.baseActivity, this.txt_sliding, "slidingControlSpeed");
        this.iplay_playView.setProgressBar(this.iplay_progress_speed);
        this.iplay_playView_vibrate.setProgressBar(this.iplay_progress_vibrate);
        Glide.with(this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.iplay_layout_bg);
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
        if (sensorEvent.sensor.getType() != 1) {
            sensorEvent.sensor.getType();
            return;
        }
        float f = sensorEvent.values[0];
        float f2 = sensorEvent.values[1];
        float f3 = sensorEvent.values[2];
        long currentTimeMillis = System.currentTimeMillis() / 1000;
        long currentTimeMillis2 = System.currentTimeMillis() / 1000;
        this.mBleSpeedUtil.shakeSpeed(f, f2, f3);
        this.mBleVibrateUtil.shakeSpeed(f, f2, f3, new BleSpeedUtil.OnShakeListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.16
            @Override // com.hismith.hismithapp.utils.BleSpeedUtil.OnShakeListener
            public void onShake(int i) {
                BleDeviceModel bleDeviceModel = BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).getmConnBleDevice();
                if (bleDeviceModel == null || i != bleDeviceModel.getVibration()) {
                    BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeVibrate(i);
                }
            }
        });
        if (f3 <= -9.0f && f3 > -10.0f) {
            int i = this.reversalCount + 1;
            this.reversalCount = i;
            if (i >= 4) {
                this.recoverCount = 0;
                if (!this.isReversal) {
                    LogUtil.e("Sensor value 翻转翻转翻转");
                    this.isReversal = true;
                    BleManager.getInstance(this.baseActivity).writeSpeed(0);
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.17
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeVibrate(0);
                        }
                    }, 100L);
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.18
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(CC_SMVPlayFragment.this.baseActivity).writeCloseVoice();
                        }
                    }, 200L);
                }
            }
        } else if (f3 > -1.0f) {
            int i2 = this.recoverCount + 1;
            this.recoverCount = i2;
            if (i2 >= 4) {
                this.reversalCount = 0;
                if (this.isReversal) {
                    LogUtil.e("Sensor value 翻转恢复");
                    this.isReversal = false;
                    if (this.mode <= 0) {
                        BleManager.getInstance(this.baseActivity).writeSpeed(this.iplay_progress_speed.getProgress());
                    } else {
                        int size = this.mModes.size();
                        int i3 = this.mode;
                        BaseActivity baseActivity = this.baseActivity;
                        if (size >= i3) {
                            BleManager.getInstance(baseActivity).writeData(this.mModes.get(this.mode - 1).getmCommand());
                        } else {
                            BleManager.getInstance(baseActivity).writeSpeedMode(this.mode);
                        }
                    }
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMVPlayFragment.19
                        @Override // java.lang.Runnable
                        public void run() {
                            int i4 = CC_SMVPlayFragment.this.vibrationMode;
                            CC_SMVPlayFragment cC_SMVPlayFragment = CC_SMVPlayFragment.this;
                            if (i4 <= 0) {
                                BleManager.getInstance(cC_SMVPlayFragment.baseActivity).writeVibrate(CC_SMVPlayFragment.this.iplay_progress_vibrate.getProgress());
                                return;
                            }
                            int size2 = cC_SMVPlayFragment.mVibrationModes.size();
                            int i5 = CC_SMVPlayFragment.this.vibrationMode;
                            CC_SMVPlayFragment cC_SMVPlayFragment2 = CC_SMVPlayFragment.this;
                            if (size2 >= i5) {
                                BleManager.getInstance(cC_SMVPlayFragment2.baseActivity).writeData(((ProductModel.Mode) CC_SMVPlayFragment.this.mVibrationModes.get(CC_SMVPlayFragment.this.vibrationMode - 1)).getmCommand());
                            } else {
                                BleManager.getInstance(cC_SMVPlayFragment2.baseActivity).writeVibrateMode(CC_SMVPlayFragment.this.vibrationMode);
                            }
                        }
                    }, 100L);
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
