package com.hismith.hismithapp.fragment.playui;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
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
import android.view.ViewTreeObserver;
import android.view.animation.LinearInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
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
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.PlayView_Four;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class CC_SMSPlayFragment extends LazyLoadFragment implements View.OnClickListener, SensorEventListener {
    private static final float NS2S = 1.0E-9f;
    private String address;
    private BaseActivity baseActivity;
    private int currentPosition;
    private int downPosition;
    private float downX;
    private float endDownX;
    private ImageView img_run_mode;
    private ImageView iplay_img_close_mode;
    private ImageView iplay_img_mic;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_position;
    private RelativeLayout iplay_layout;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
    private ImageView iplay_layout_bg;
    private FrameLayout iplay_layout_mic_btn;
    private FrameLayout iplay_layout_mode;
    private FrameLayout iplay_layout_mode_btn;
    private FrameLayout iplay_layout_touch_position;
    private PlayView_Four iplay_playView;
    private RecyclerView iplay_recycler_mode;
    private ConstraintLayout layout_position_max;
    private ConstraintLayout layout_position_min;
    private FrameLayout layout_speed_max;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private BleSpeedUtil mBleSpeedUtil;
    private GridLayoutManager mLayoutManager;
    private List<ProductModel.Mode> mModes;
    private int mPosition;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private ProgressBar progress_position;
    private ProgressBar progress_speed;
    private RecorderVolumeManager recorderVolumeManager;
    private CustomSeekbar seek_smooth;
    private float startDownX;
    private float timestamp;
    private TextView txt_position;
    private TextView txt_position_label;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private TextView txt_smooth;
    private TextView txt_smooth_label;
    private TextView txt_speed;
    private TextView txt_speed_label;
    private TextView txt_speed_max;
    private View view_touch;
    private float[] angle = new float[3];
    private boolean isReversal = false;
    private int mMode = -1;
    private int mSpeed = 0;
    private int start_stop_position_diff = Constans.start_stop_position_diff;
    private boolean runMode = false;
    private boolean runPisitionQiu = false;
    private boolean isMicMode = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            Bundle data = message.getData();
            String string = data != null ? data.getString(Constans.BleType) : "";
            int i = message.what;
            if (i == 1) {
                int intValue = ((Integer) message.obj).intValue();
                if (intValue == 160) {
                    CC_SMSPlayFragment.this.runMode = true;
                    CC_SMSPlayFragment.this.runPisitionQiu = false;
                    CC_SMSPlayFragment.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_on);
                    CC_SMSPlayFragment.this.iplay_layout_touch_position.setVisibility(8);
                    CC_SMSPlayFragment.this.iplay_playView.setTouch(true);
                } else if (intValue == 161) {
                    CC_SMSPlayFragment.this.runMode = false;
                    CC_SMSPlayFragment.this.runPisitionQiu = true;
                    CC_SMSPlayFragment.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_off);
                    CC_SMSPlayFragment.this.iplay_layout_touch_position.setVisibility(0);
                    CC_SMSPlayFragment.this.iplay_playView.setTouch(false);
                    CC_SMSPlayFragment.this.manualTouchPosition();
                }
            } else if (i == 3) {
                CC_SMSPlayFragment.this.refreshSpeedUI(((Integer) message.obj).intValue());
            } else if (i != 4) {
                switch (i) {
                    case 7:
                        CC_SMSPlayFragment.this.refreshStartBitUI(((Integer) message.obj).intValue());
                        "CC".equals(string);
                        break;
                    case 8:
                        CC_SMSPlayFragment.this.refreshEndBitUI(((Integer) message.obj).intValue());
                        "CC".equals(string);
                        break;
                    case 9:
                        CC_SMSPlayFragment.this.refreshSmoothUI(((Integer) message.obj).intValue());
                        break;
                    case 10:
                        CC_SMSPlayFragment.this.refreshPositionUI(((Integer) message.obj).intValue());
                        if (CC_SMSPlayFragment.this.runPisitionQiu) {
                            CC_SMSPlayFragment.this.manualTouchPosition();
                            break;
                        }
                        break;
                }
            } else {
                CC_SMSPlayFragment.this.refreshSpeedModeUI(((Integer) message.obj).intValue());
            }
            return false;
        }
    });
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private int mSafeMax = 100;
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.12
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                CC_SMSPlayFragment.this.downX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - CC_SMSPlayFragment.this.downX;
            int right = CC_SMSPlayFragment.this.progress_speed.getRight() + (CC_SMSPlayFragment.this.layout_speed_max.getWidth() / 2);
            LogUtil.e("getRight():" + CC_SMSPlayFragment.this.layout_speed_max.getRight() + ", width:" + CC_SMSPlayFragment.this.progress_speed.getRight() + ", " + CC_SMSPlayFragment.this.layout_speed_max.getWidth());
            int right2 = (int) (right - (CC_SMSPlayFragment.this.layout_speed_max.getRight() + x));
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > CC_SMSPlayFragment.this.progress_speed.getWidth()) {
                right2 = CC_SMSPlayFragment.this.progress_speed.getWidth();
            }
            LogUtil.e("safe right:" + right2);
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) CC_SMSPlayFragment.this.layout_speed_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            CC_SMSPlayFragment.this.layout_speed_max.setLayoutParams(layoutParams);
            int width = (int) (100.0d - ((right2 * 100.0d) / CC_SMSPlayFragment.this.progress_speed.getWidth()));
            CC_SMSPlayFragment.this.mSafeMax = width;
            CC_SMSPlayFragment.this.txt_speed_max.setText(width + "");
            SharedPreferencesUtil.put(CC_SMSPlayFragment.this.baseActivity, Constans.SAFE_FUN1, Integer.valueOf(CC_SMSPlayFragment.this.mSafeMax));
            if (width < 0 || width >= CC_SMSPlayFragment.this.mSpeed) {
                return true;
            }
            BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writeSpeed(width);
            CC_SMSPlayFragment.this.mSpeed = width;
            CC_SMSPlayFragment.this.progress_speed.setProgress(width);
            CC_SMSPlayFragment.this.txt_speed.setText(width + "");
            return true;
        }
    };
    private int mPositionEnd = 100;
    private View.OnTouchListener endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.13
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                CC_SMSPlayFragment.this.endDownX = motionEvent.getX();
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - CC_SMSPlayFragment.this.endDownX;
            int width = CC_SMSPlayFragment.this.progress_position.getWidth();
            int right = (int) ((CC_SMSPlayFragment.this.progress_position.getRight() + (CC_SMSPlayFragment.this.layout_position_max.getWidth() / 2)) - (CC_SMSPlayFragment.this.layout_position_max.getRight() + x));
            LogUtil.e("onTouch moveX:" + x + ", right():" + right);
            if (right < 0) {
                right = 0;
            } else if (right > width) {
                right = width;
            }
            double d = width;
            int i = (int) (100.0d - ((right * 100.0d) / d));
            if (i <= CC_SMSPlayFragment.this.mPositionStart + CC_SMSPlayFragment.this.start_stop_position_diff) {
                int i2 = CC_SMSPlayFragment.this.mPositionStart + CC_SMSPlayFragment.this.start_stop_position_diff;
                i = i2 > 100 ? 100 : i2;
                right = (int) (d - ((d / 100.0d) * i));
            }
            int i3 = i <= 100 ? i < 0 ? 0 : i : 100;
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) CC_SMSPlayFragment.this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right, 0);
            CC_SMSPlayFragment.this.layout_position_max.setLayoutParams(layoutParams);
            if (CC_SMSPlayFragment.this.mPositionEnd == i3) {
                return true;
            }
            CC_SMSPlayFragment.this.mPositionEnd = i3;
            LogUtil.e("结束位置：" + CC_SMSPlayFragment.this.mPositionEnd + ", right:" + right);
            CC_SMSPlayFragment.this.txt_position_max.setText(i3 + "");
            BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writeStopPosition(CC_SMSPlayFragment.this.mPositionEnd);
            return true;
        }
    };
    private int mPositionStart = 0;
    private View.OnTouchListener startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.14
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                CC_SMSPlayFragment.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - CC_SMSPlayFragment.this.startDownX;
            LogUtil.e("抖动：" + motionEvent.getX() + ", " + CC_SMSPlayFragment.this.startDownX + " = " + x);
            int width = CC_SMSPlayFragment.this.progress_position.getWidth();
            int left = (int) ((CC_SMSPlayFragment.this.layout_position_min.getLeft() + x) - (CC_SMSPlayFragment.this.progress_position.getLeft() - (CC_SMSPlayFragment.this.layout_position_min.getWidth() / 2)));
            LogUtil.e("onTouch moveX:" + x + ", left():" + left);
            if (left < 0) {
                left = 0;
            } else if (left > width) {
                left = width;
            }
            double d = width;
            int i = (int) ((left * 100.0d) / d);
            if (i >= CC_SMSPlayFragment.this.mPositionEnd - CC_SMSPlayFragment.this.start_stop_position_diff) {
                int i2 = CC_SMSPlayFragment.this.mPositionEnd - CC_SMSPlayFragment.this.start_stop_position_diff;
                i = i2 < 0 ? 0 : i2;
                left = (int) ((d / 100.0d) * i);
            }
            if (i > 100) {
                i = 100;
            } else if (i < 0) {
                i = 0;
            }
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) CC_SMSPlayFragment.this.layout_position_min.getLayoutParams();
            layoutParams.setMargins(left, 0, 0, 0);
            CC_SMSPlayFragment.this.layout_position_min.setLayoutParams(layoutParams);
            if (CC_SMSPlayFragment.this.mPositionStart == i) {
                return true;
            }
            CC_SMSPlayFragment.this.mPositionStart = i;
            LogUtil.e("开始位置：" + CC_SMSPlayFragment.this.mPositionStart + ", left:" + left);
            CC_SMSPlayFragment.this.txt_position_min.setText(i + "");
            BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writeStartPosition(CC_SMSPlayFragment.this.mPositionStart);
            return true;
        }
    };
    private View.OnTouchListener positionOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.16
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            int i = 0;
            if (action == 0) {
                CC_SMSPlayFragment.this.runPisitionQiu = false;
            } else if (action != 2) {
                return true;
            }
            CC_SMSPlayFragment cC_SMSPlayFragment = CC_SMSPlayFragment.this;
            int changeUIPosition = cC_SMSPlayFragment.changeUIPosition(cC_SMSPlayFragment.iplay_layout_touch_position, CC_SMSPlayFragment.this.iplay_img_position, motionEvent.getX(), motionEvent.getY());
            LogUtil.e("x:" + motionEvent.getX() + ", y:" + motionEvent.getY() + ", top：" + CC_SMSPlayFragment.this.iplay_img_position.getTop());
            if (changeUIPosition > 100) {
                i = 100;
            } else if (changeUIPosition >= 0) {
                i = changeUIPosition;
            }
            BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writePosition(i);
            return true;
        }
    };
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.17
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                CC_SMSPlayFragment.this.initProductData();
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
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    CC_SMSPlayFragment.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public int changeUIPosition(View view, View view2, float f, float f2) {
        int width = (int) (f - (view2.getWidth() / 2.0f));
        int height = (int) (f2 - (view2.getHeight() / 2.0f));
        int width2 = view2.getWidth() + width;
        int height2 = view2.getHeight() + height;
        if (width < 0) {
            width2 = view2.getWidth();
            width = 0;
        }
        if (width2 > view.getWidth()) {
            width2 = view.getWidth();
            width = width2 - view2.getWidth();
        }
        if (height < 0) {
            height2 = view2.getHeight();
            height = 0;
        }
        if (height2 > view.getHeight()) {
            height2 = view.getHeight();
            height = height2 - view2.getHeight();
        }
        view2.layout(width, height, width2, height2);
        return 100 - ((int) (((height * 1.0f) / (this.iplay_layout_touch_position.getHeight() - view2.getHeight())) * 100.0f));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideGestureTip() {
    }

    private void init() {
        this.recorderVolumeManager = new RecorderVolumeManager();
        initSensor();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.iplay_recycler_mode.setLayoutManager(gridLayoutManager);
        this.mModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = iplayModeAdapter;
        iplayModeAdapter.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.3
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) CC_SMSPlayFragment.this.mModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writeData(str);
            }
        });
        this.iplay_recycler_mode.setAdapter(this.mAdapter);
        this.mSafeMax = ((Integer) SharedPreferencesUtil.get(this.baseActivity, Constans.SAFE_FUN1, 40)).intValue();
        this.layout_speed_max.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.4
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                CC_SMSPlayFragment.this.layout_speed_max.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                CC_SMSPlayFragment cC_SMSPlayFragment = CC_SMSPlayFragment.this;
                cC_SMSPlayFragment.refreshSafeUI(cC_SMSPlayFragment.mSafeMax);
            }
        });
        initBleDataUI();
        initProductData();
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2, String str3) {
        try {
            LogUtil.e("data0:" + str2 + ", data1:" + str3);
            Message message = new Message();
            message.what = Integer.valueOf(str2, 16).intValue();
            message.obj = Integer.valueOf(str3, 16);
            Bundle bundle = new Bundle();
            bundle.putString(Constans.BleType, str);
            message.setData(bundle);
            this.mHandler.sendMessage(message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initBleDataUI() {
        if (BleManager.getInstance(this.baseActivity).getmConnBleDevice() != null) {
            Message message = new Message();
            message.what = 1;
            message.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getRunMode() == 0 ? 160 : 161);
            this.mHandler.sendMessage(message);
            Message message2 = new Message();
            message2.what = 3;
            message2.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed());
            this.mHandler.sendMessage(message2);
            Message message3 = new Message();
            message3.what = 4;
            message3.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMode());
            this.mHandler.sendMessage(message3);
            int startPosition = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStartPosition();
            Message message4 = new Message();
            message4.what = 7;
            message4.obj = Integer.valueOf(startPosition);
            this.mHandler.sendMessage(message4);
            int stopPosition = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStopPosition();
            Message message5 = new Message();
            message5.what = 8;
            if (stopPosition <= 0) {
                stopPosition = this.start_stop_position_diff + startPosition;
            }
            message5.obj = Integer.valueOf(stopPosition);
            this.mHandler.sendMessage(message5);
            Message message6 = new Message();
            message6.what = 9;
            message6.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSmoothness());
            this.mHandler.sendMessage(message6);
            Message message7 = new Message();
            message7.what = 10;
            message7.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getPosition());
            this.mHandler.sendMessage(message7);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initModeShow() {
        int i = this.isShowMode;
        if (i != 0) {
            if (i == 1) {
                startAnimation(this.iplay_layout_mode);
            }
        } else {
            FrameLayout frameLayout = this.iplay_layout_mode;
            if (frameLayout == null || frameLayout.getVisibility() != 0) {
                return;
            }
            stopAnimation(this.iplay_layout_mode);
        }
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
        this.mModes.clear();
        boolean hasMode = ProductUtil.hasMode(0);
        FrameLayout frameLayout = this.iplay_layout_mode_btn;
        if (hasMode) {
            frameLayout.setVisibility(0);
            this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
        } else {
            frameLayout.setVisibility(8);
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
    }

    private void initRecordPermission() {
        if (ContextCompat.checkSelfPermission(this.baseActivity, "android.permission.RECORD_AUDIO") == 0) {
            startMicClick();
            return;
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                CC_SMSPlayFragment.this.m603x7d2df1f2((Permission) obj);
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

    /* JADX INFO: Access modifiers changed from: private */
    public void manualTouchPosition() {
        int position = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getPosition();
        float stopPosition = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStopPosition() - BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStartPosition();
        final float x = this.iplay_img_position.getX() + (this.iplay_img_position.getWidth() / 2.0f);
        final float height = (((this.iplay_layout_touch_position.getHeight() - this.iplay_img_position.getHeight()) / stopPosition) * (stopPosition - position)) + (this.iplay_img_position.getHeight() / 2.0f);
        this.iplay_img_position.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.18
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                CC_SMSPlayFragment.this.iplay_img_position.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                CC_SMSPlayFragment cC_SMSPlayFragment = CC_SMSPlayFragment.this;
                cC_SMSPlayFragment.changeUIPosition(cC_SMSPlayFragment.iplay_layout_touch_position, CC_SMSPlayFragment.this.iplay_img_position, x, height);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        if (this.mPositionEnd != i) {
            this.mPositionEnd = i;
            this.txt_position_max.setText(this.mPositionEnd + "");
            double width = this.progress_position.getWidth();
            int i2 = (int) (width - ((width / 100.0d) * i));
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, i2, 0);
            this.layout_position_max.setLayoutParams(layoutParams);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionUI(int i) {
        this.mPosition = i;
        this.txt_position.setText(i + "");
        this.progress_position.setProgress(i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.txt_speed_max.setText(i + "");
        double width = this.progress_speed.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_speed_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_speed_max.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSmoothUI(int i) {
        this.txt_smooth.setText(i + "");
        ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.15
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                CC_SMSPlayFragment.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
            }
        });
        ofInt.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedModeUI(int i) {
        ImageView imageView = this.iplay_img_mode;
        if (i <= 0) {
            imageView.setImageResource(R.mipmap.ic_iplay_mode);
            this.isShowMode = 0;
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
            }
        } else {
            imageView.setImageResource(R.mipmap.ic_iplay_mode_select);
            this.isShowMode = 1;
            if (this.iplay_layout_mode.getVisibility() != 0) {
                startAnimation(this.iplay_layout_mode);
            }
        }
        IplayModeAdapter iplayModeAdapter = this.mAdapter;
        if (iplayModeAdapter != null) {
            iplayModeAdapter.setClickPosition(i - 1);
            if (i != 0) {
                this.iplay_recycler_mode.smoothScrollToPosition(i);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedUI(int i) {
        this.progress_speed.setProgress(i);
        this.txt_speed.setText(i + "");
        refreshSpeedModeUI(0);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartBitUI(int i) {
        if (this.mPositionStart != i) {
            this.mPositionStart = i;
            this.txt_position_min.setText(this.mPositionStart + "");
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_min.getLayoutParams();
            layoutParams.setMargins((int) ((this.progress_position.getWidth() / 100.0d) * i), 0, 0, 0);
            this.layout_position_min.setLayoutParams(layoutParams);
        }
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

    private void setListener() {
        this.img_run_mode.setOnClickListener(this);
        this.iplay_layout_mic_btn.setOnClickListener(this);
        this.iplay_layout_mode_btn.setOnClickListener(this);
        this.iplay_img_close_mode.setOnClickListener(this);
        this.layout_speed_max.setOnTouchListener(this.safeOnTouch);
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
        this.iplay_layout_touch_position.setOnTouchListener(this.positionOnTouch);
        this.iplay_playView.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.5
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                if (i < 0) {
                    i = 0;
                } else if (i > 100) {
                    i = 100;
                }
                if (i > CC_SMSPlayFragment.this.mSafeMax) {
                    i = CC_SMSPlayFragment.this.mSafeMax;
                }
                if (i != CC_SMSPlayFragment.this.progress_speed.getProgress() || i == 0) {
                    CC_SMSPlayFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (CC_SMSPlayFragment.this.mMode != 0) {
                        CC_SMSPlayFragment.this.mMode = 0;
                        if (CC_SMSPlayFragment.this.mAdapter != null) {
                            CC_SMSPlayFragment.this.mAdapter.setClickPosition(CC_SMSPlayFragment.this.mMode - 1);
                        }
                    }
                    BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writeSpeed(i);
                    CC_SMSPlayFragment.this.mSpeed = i;
                }
            }
        });
        this.view_touch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.6
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                CC_SMSPlayFragment.this.hideGestureTip();
                return false;
            }
        });
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.7
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
                if (i > CC_SMSPlayFragment.this.mSafeMax) {
                    i = CC_SMSPlayFragment.this.mSafeMax;
                }
                CC_SMSPlayFragment.this.mSpeed = i;
                BleDeviceModel bleDeviceModel = BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).getmConnBleDevice();
                if (bleDeviceModel == null || i != bleDeviceModel.getSpeed()) {
                    BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writeSpeed(i);
                }
            }
        });
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.8
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                CC_SMSPlayFragment.this.scrollModeHideImg();
            }
        });
    }

    private void startAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 0.0f, 1.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.10
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                CC_SMSPlayFragment.this.inAnim = false;
                if (i != CC_SMSPlayFragment.this.isShowMode) {
                    CC_SMSPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                CC_SMSPlayFragment.this.inAnim = true;
                frameLayout.setVisibility(0);
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    private void startMicClick() {
        boolean z = this.isMicMode;
        this.isMicMode = !z;
        ImageView imageView = this.iplay_img_mic;
        if (!z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic_select);
            this.recorderVolumeManager.startRecording();
        } else {
            imageView.setImageResource(R.mipmap.ic_iplay_mic);
            this.recorderVolumeManager.stopRecording();
            BleManager.getInstance(this.baseActivity).writeSpeed(0);
        }
    }

    private void stopAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 1.0f, 0.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.11
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                CC_SMSPlayFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != CC_SMSPlayFragment.this.isShowMode) {
                    CC_SMSPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                CC_SMSPlayFragment.this.inAnim = true;
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

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-fragment-playui-CC_SMSPlayFragment, reason: not valid java name */
    /* synthetic */ void m603x7d2df1f2(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            startMicClick();
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.9
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + CC_SMSPlayFragment.this.baseActivity.getPackageName()));
                    CC_SMSPlayFragment.this.startActivity(intent);
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
        if (view.getId() == R.id.img_run_mode) {
            if (this.runMode) {
                Message message = new Message();
                message.what = 1;
                message.obj = 161;
                this.mHandler.sendMessage(message);
                BleManager.getInstance(this.baseActivity).writeRunManual();
                return;
            }
            Message message2 = new Message();
            message2.what = 1;
            message2.obj = 160;
            this.mHandler.sendMessage(message2);
            BleManager.getInstance(this.baseActivity).writeRunAuto();
            return;
        }
        if (view.getId() == R.id.iplay_layout_mode_btn) {
            if (this.inAnim) {
                return;
            }
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
                return;
            } else {
                this.isShowMode = 1;
                startAnimation(this.iplay_layout_mode);
                return;
            }
        }
        if (view.getId() == R.id.iplay_layout_mic_btn) {
            initRecordPermission();
            return;
        }
        if (view.getId() == R.id.iplay_img_close_mode) {
            BleManager.getInstance(this.baseActivity).writeSpeed(this.progress_speed.getProgress());
            this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
            IplayModeAdapter iplayModeAdapter = this.mAdapter;
            if (iplayModeAdapter != null) {
                iplayModeAdapter.setClickPosition(-1);
            }
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_cc_sms, (ViewGroup) null);
        this.iplay_layout = (RelativeLayout) inflate.findViewById(R.id.iplay_layout);
        this.iplay_layout_bg = (ImageView) inflate.findViewById(R.id.iplay_layout_bg);
        this.img_run_mode = (ImageView) inflate.findViewById(R.id.img_run_mode);
        this.iplay_layout_mic_btn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mic_btn);
        this.iplay_img_mic = (ImageView) inflate.findViewById(R.id.iplay_img_mic);
        this.iplay_layout_mode_btn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode_btn);
        this.iplay_img_mode = (ImageView) inflate.findViewById(R.id.iplay_img_mode);
        this.txt_speed_label = (TextView) inflate.findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) inflate.findViewById(R.id.txt_speed);
        this.progress_speed = (ProgressBar) inflate.findViewById(R.id.progress_speed);
        this.layout_speed_max = (FrameLayout) inflate.findViewById(R.id.layout_speed_max);
        this.txt_speed_max = (TextView) inflate.findViewById(R.id.txt_speed_max);
        this.txt_position_label = (TextView) inflate.findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) inflate.findViewById(R.id.txt_position);
        this.progress_position = (ProgressBar) inflate.findViewById(R.id.progress_position);
        this.layout_position_max = (ConstraintLayout) inflate.findViewById(R.id.layout_position_max);
        this.txt_position_max = (TextView) inflate.findViewById(R.id.txt_position_max);
        this.layout_position_min = (ConstraintLayout) inflate.findViewById(R.id.layout_position_min);
        this.txt_position_min = (TextView) inflate.findViewById(R.id.txt_position_min);
        this.txt_smooth_label = (TextView) inflate.findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) inflate.findViewById(R.id.txt_smooth);
        this.seek_smooth = (CustomSeekbar) inflate.findViewById(R.id.seek_smooth);
        this.iplay_playView = (PlayView_Four) inflate.findViewById(R.id.iplay_playView);
        this.iplay_layout_touch_position = (FrameLayout) inflate.findViewById(R.id.iplay_layout_touch_position);
        this.iplay_img_position = (ImageView) inflate.findViewById(R.id.iplay_img_position);
        this.iplay_layout_mode = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode);
        this.iplay_recycler_mode = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_mode);
        this.iplay_layout_arrow_up = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_down);
        this.iplay_img_close_mode = (ImageView) inflate.findViewById(R.id.iplay_img_close_mode);
        this.view_touch = inflate.findViewById(R.id.view_touch);
        LanguageUtil.setText(this.baseActivity, this.txt_speed_label, "speedTitle");
        this.txt_position_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "position") + ":");
        this.txt_smooth_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "smoothness") + ":");
        this.iplay_playView.setProgressBar(this.progress_speed);
        this.iplay_playView.setTouch(false);
        Glide.with(this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.iplay_layout_bg);
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.2
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + CC_SMSPlayFragment.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + CC_SMSPlayFragment.this.seek_smooth.getProgress());
                int progress = CC_SMSPlayFragment.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                int i3 = i / 100;
                CC_SMSPlayFragment.this.txt_smooth.setText(i3 + "");
                BleManager.getInstance(CC_SMSPlayFragment.this.baseActivity).writeSmooth(i3);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.playui.CC_SMSPlayFragment.2.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        CC_SMSPlayFragment.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
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
        if (this.mSensor != null) {
            this.mSensorManager.unregisterListener(this);
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
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
            if (f3 <= -9.0f && f3 > -10.0f) {
                int i = this.reversalCount + 1;
                this.reversalCount = i;
                if (i >= 4) {
                    this.recoverCount = 0;
                    if (!this.isReversal && this.runMode) {
                        LogUtil.e("Sensor value 翻转翻转翻转");
                        this.isReversal = true;
                        BleManager.getInstance(this.baseActivity).writeSpeed(0);
                    }
                }
            } else if (f3 > -1.0f) {
                int i2 = this.recoverCount + 1;
                this.recoverCount = i2;
                if (i2 >= 4) {
                    this.reversalCount = 0;
                    if (this.isReversal && this.runMode) {
                        LogUtil.e("Sensor value 翻转恢复");
                        this.isReversal = false;
                        if (this.mMode <= 0) {
                            BleManager.getInstance(this.baseActivity).writeSpeed(this.progress_speed.getProgress());
                        } else {
                            int size = this.mModes.size();
                            int i3 = this.mMode;
                            BaseActivity baseActivity = this.baseActivity;
                            if (size >= i3) {
                                BleManager.getInstance(baseActivity).writeData(this.mModes.get(this.mMode - 1).getmCommand());
                            } else {
                                BleManager.getInstance(baseActivity).writeSpeedMode(this.mMode);
                            }
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
