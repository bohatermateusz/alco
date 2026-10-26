package com.hismith.hismithapp;

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
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.view.animation.LinearInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
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
public class IPlayNewActivity extends BaseActivity implements View.OnClickListener, BleManager.OnConnectStateListener, SensorEventListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private static final float NS2S = 1.0E-9f;
    private int currentPosition;
    private int downPosition;
    private float downX;
    private float endDownX;
    private ImageView img_close_left;
    private ImageView img_close_right;
    private ImageView img_gesture_left;
    private ImageView img_gesture_right;
    private ImageView img_run_mode;
    private ImageView iplay_img_close_mode;
    private ImageView iplay_img_close_vibration;
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
    private FrameLayout iplay_layout_vibrate_arrow_down;
    private FrameLayout iplay_layout_vibrate_arrow_up;
    private FrameLayout iplay_layout_vibration;
    private PlayView_Four iplay_playView;
    private RecyclerView iplay_recycler_mode;
    private RecyclerView iplay_recycler_vibration;
    private FrameLayout layout_position_max;
    private FrameLayout layout_position_min;
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
    private TextView title_txt_status;
    private TextView txt_gesture_tip;
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
    private boolean isMicMode = false;
    private int leftAction = 1;
    private int rightAction = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.IPlayNewActivity.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int intValue = ((Integer) message.obj).intValue();
            Bundle data = message.getData();
            String string = data != null ? data.getString(Constans.BleType) : "";
            int i = message.what;
            if (i != 1) {
                if (i == 3) {
                    IPlayNewActivity.this.refreshSpeedUI(intValue);
                } else if (i != 4) {
                    switch (i) {
                        case 7:
                            if (!"CC".equals(string)) {
                                IPlayNewActivity.this.refreshStartBitUI(intValue);
                                break;
                            }
                            break;
                        case 8:
                            if (!"CC".equals(string)) {
                                IPlayNewActivity.this.refreshEndBitUI(intValue);
                                break;
                            }
                            break;
                        case 9:
                            IPlayNewActivity.this.refreshSmoothUI(intValue);
                            break;
                        case 10:
                            IPlayNewActivity.this.refreshPositionUI(intValue);
                            break;
                    }
                } else {
                    IPlayNewActivity.this.refreshSpeedModeUI(intValue);
                }
            } else if (intValue == 160) {
                IPlayNewActivity.this.runMode = true;
                IPlayNewActivity.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_on);
                IPlayNewActivity.this.iplay_layout_touch_position.setVisibility(8);
                IPlayNewActivity.this.iplay_playView.setTouch(true);
            } else if (intValue == 161) {
                IPlayNewActivity.this.runMode = false;
                IPlayNewActivity.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_off);
                IPlayNewActivity.this.iplay_layout_touch_position.setVisibility(0);
                IPlayNewActivity.this.iplay_playView.setTouch(false);
            }
            return false;
        }
    });
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.12
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayNewActivity.this.leftAction = 0;
                IPlayNewActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayNewActivity.this.lastLeftClickTime;
                IPlayNewActivity iPlayNewActivity = IPlayNewActivity.this;
                if (j < 500) {
                    iPlayNewActivity.finish();
                    return true;
                }
                iPlayNewActivity.lastLeftClickTime = currentTimeMillis;
                IPlayNewActivity.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.13
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayNewActivity.this.rightAction = 0;
                IPlayNewActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - IPlayNewActivity.this.lastRightClickTime;
                IPlayNewActivity iPlayNewActivity = IPlayNewActivity.this;
                if (j < 500) {
                    iPlayNewActivity.finish();
                    return true;
                }
                iPlayNewActivity.lastRightClickTime = currentTimeMillis;
                IPlayNewActivity.this.rightAction = 1;
            }
            return true;
        }
    };
    private int i = 0;
    private int mSafeMax = 100;
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.14
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayNewActivity.this.downX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - IPlayNewActivity.this.downX;
            int right = IPlayNewActivity.this.progress_speed.getRight() + (IPlayNewActivity.this.layout_speed_max.getWidth() / 2);
            LogUtil.e("getRight():" + IPlayNewActivity.this.layout_speed_max.getRight() + ", width:" + IPlayNewActivity.this.progress_speed.getRight() + ", " + IPlayNewActivity.this.layout_speed_max.getWidth());
            int right2 = (int) (right - (IPlayNewActivity.this.layout_speed_max.getRight() + x));
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > IPlayNewActivity.this.progress_speed.getWidth()) {
                right2 = IPlayNewActivity.this.progress_speed.getWidth();
            }
            LogUtil.e("safe right:" + right2);
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) IPlayNewActivity.this.layout_speed_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            IPlayNewActivity.this.layout_speed_max.setLayoutParams(layoutParams);
            int width = (int) (100.0d - ((right2 * 100.0d) / IPlayNewActivity.this.progress_speed.getWidth()));
            IPlayNewActivity.this.mSafeMax = width;
            IPlayNewActivity.this.txt_speed_max.setText(width + "");
            if (width < 0 || width >= IPlayNewActivity.this.mSpeed) {
                return true;
            }
            BleManager.getInstance(IPlayNewActivity.this).writeSpeed(width);
            IPlayNewActivity.this.mSpeed = width;
            IPlayNewActivity.this.progress_speed.setProgress(width);
            IPlayNewActivity.this.txt_speed.setText(width + "");
            return true;
        }
    };
    private int mPositionEnd = 100;
    private View.OnTouchListener endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.15
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                IPlayNewActivity.this.endDownX = motionEvent.getX();
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - IPlayNewActivity.this.endDownX;
            int right = IPlayNewActivity.this.progress_position.getRight() + (IPlayNewActivity.this.layout_position_max.getWidth() / 2);
            LogUtil.e("getRight:" + IPlayNewActivity.this.layout_position_max.getRight());
            int right2 = (int) (right - (IPlayNewActivity.this.layout_position_max.getRight() + x));
            LogUtil.e("onTouch moveX:" + x + ", right():" + right2);
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > IPlayNewActivity.this.progress_position.getWidth()) {
                right2 = IPlayNewActivity.this.progress_position.getWidth();
            }
            int width = (int) (100.0d - ((right2 * 100.0d) / IPlayNewActivity.this.progress_position.getWidth()));
            if (width <= IPlayNewActivity.this.mPositionStart + IPlayNewActivity.this.start_stop_position_diff) {
                int i = IPlayNewActivity.this.mPositionStart + IPlayNewActivity.this.start_stop_position_diff;
                width = i > 100 ? 100 : i;
                right2 = (int) (IPlayNewActivity.this.progress_position.getWidth() - ((IPlayNewActivity.this.progress_position.getWidth() / 100.0d) * width));
            }
            int i2 = width <= 100 ? width < 0 ? 0 : width : 100;
            RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) IPlayNewActivity.this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            IPlayNewActivity.this.layout_position_max.setLayoutParams(layoutParams);
            IPlayNewActivity.this.mPositionEnd = i2;
            LogUtil.e("结束位置：" + IPlayNewActivity.this.mPositionEnd + ", right:" + right2);
            IPlayNewActivity.this.txt_position_max.setText(i2 + "");
            BleManager.getInstance(IPlayNewActivity.this).writeStopPosition(IPlayNewActivity.this.mPositionEnd);
            return true;
        }
    };
    private int mPositionStart = 0;
    private View.OnTouchListener startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.17
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                IPlayNewActivity.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - IPlayNewActivity.this.startDownX;
            int width = IPlayNewActivity.this.progress_position.getWidth();
            int left = (int) (IPlayNewActivity.this.layout_position_min.getLeft() + x);
            LogUtil.e("onTouch moveX:" + x + ", left():" + left);
            if (left < 0) {
                left = 0;
            } else if (left > width) {
                left = width;
            }
            double d = width;
            int i = (int) ((left * 100.0d) / d);
            if (i >= IPlayNewActivity.this.mPositionEnd - IPlayNewActivity.this.start_stop_position_diff) {
                i = IPlayNewActivity.this.mPositionEnd - IPlayNewActivity.this.start_stop_position_diff;
                if (i < 0) {
                    i = 0;
                }
                left = (int) ((d / 100.0d) * i);
            }
            if (i > 100) {
                i = 100;
            } else if (i < 0) {
                i = 0;
            }
            RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) IPlayNewActivity.this.layout_position_min.getLayoutParams();
            layoutParams.setMargins(left, 0, 0, 0);
            IPlayNewActivity.this.layout_position_min.setLayoutParams(layoutParams);
            IPlayNewActivity.this.mPositionStart = i;
            LogUtil.e("开始位置：" + IPlayNewActivity.this.mPositionStart + ", left:" + left);
            IPlayNewActivity.this.txt_position_min.setText(i + "");
            BleManager.getInstance(IPlayNewActivity.this).writeStartPosition(IPlayNewActivity.this.mPositionStart);
            return true;
        }
    };
    private View.OnTouchListener positionOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.18
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                IPlayNewActivity iPlayNewActivity = IPlayNewActivity.this;
                iPlayNewActivity.downPosition = iPlayNewActivity.changeUIPosition(iPlayNewActivity.iplay_layout_touch_position, IPlayNewActivity.this.iplay_img_position, motionEvent.getX(), motionEvent.getY());
                IPlayNewActivity iPlayNewActivity2 = IPlayNewActivity.this;
                iPlayNewActivity2.currentPosition = BleManager.getInstance(iPlayNewActivity2).getmConnBleDevice().getWritePosition();
            } else if (action != 2) {
                return true;
            }
            IPlayNewActivity iPlayNewActivity3 = IPlayNewActivity.this;
            int changeUIPosition = iPlayNewActivity3.changeUIPosition(iPlayNewActivity3.iplay_layout_touch_position, IPlayNewActivity.this.iplay_img_position, motionEvent.getX(), motionEvent.getY());
            IPlayNewActivity.access$4512(IPlayNewActivity.this, changeUIPosition - IPlayNewActivity.this.downPosition);
            int i = IPlayNewActivity.this.currentPosition;
            IPlayNewActivity iPlayNewActivity4 = IPlayNewActivity.this;
            if (i > 100) {
                iPlayNewActivity4.currentPosition = 100;
            } else if (iPlayNewActivity4.currentPosition < 0) {
                IPlayNewActivity.this.currentPosition = 0;
            }
            BleManager.getInstance(IPlayNewActivity.this).writePosition(IPlayNewActivity.this.currentPosition);
            LogUtil.e("downPosition:" + IPlayNewActivity.this.downPosition + ", ");
            IPlayNewActivity.this.downPosition = changeUIPosition;
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.IPlayNewActivity.21
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                IPlayNewActivity.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    IPlayNewActivity.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };
    private long triggerTime = 0;
    private long triggerLastTime = 0;
    private int shakeCount = 0;
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;

    static /* synthetic */ int access$4512(IPlayNewActivity iPlayNewActivity, int i) {
        int i2 = iPlayNewActivity.currentPosition + i;
        iPlayNewActivity.currentPosition = i2;
        return i2;
    }

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
    public void exitIPlay() {
        if (this.leftAction == 0 && this.rightAction == 0) {
            finish();
        }
        if (this.leftAction == 1) {
            LogUtil.e("onTouch 左抖动");
            shakeAnimation(this.img_close_left);
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            LogUtil.e("onTouch 右抖动");
            shakeAnimation(this.img_close_right);
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "exitDoubleGameNote"));
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideGestureTip() {
        if (this.img_gesture_left.getVisibility() == 0) {
            this.img_gesture_left.clearAnimation();
            this.img_gesture_left.setVisibility(8);
        }
        if (this.img_gesture_right.getVisibility() == 0) {
            this.img_gesture_right.clearAnimation();
            this.img_gesture_right.setVisibility(8);
        }
        if (this.txt_gesture_tip.getVisibility() == 0) {
            this.txt_gesture_tip.setVisibility(8);
        }
    }

    private void init() {
        this.recorderVolumeManager = new RecorderVolumeManager();
        initSensor();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 2);
        this.mLayoutManager = gridLayoutManager;
        this.iplay_recycler_mode.setLayoutManager(gridLayoutManager);
        this.mModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this, this.mModes);
        this.mAdapter = iplayModeAdapter;
        iplayModeAdapter.setOnItemClickListener(new IplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.4
            @Override // com.hismith.hismithapp.adapter.IplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                String str = ((ProductModel.Mode) IPlayNewActivity.this.mModes.get(i)).getmCommand();
                if (TextUtils.isEmpty(str)) {
                    return;
                }
                BleManager.getInstance(IPlayNewActivity.this).writeData(str);
            }
        });
        this.iplay_recycler_mode.setAdapter(this.mAdapter);
        BleManager.getInstance(this).initWriteMessage();
        AnimUtil.translate(this, this.img_gesture_left);
        AnimUtil.translate(this, this.img_gesture_right);
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
        if (BleManager.getInstance(this).getmConnBleDevice() != null) {
            Message message = new Message();
            message.what = 1;
            message.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getRunMode() == 0 ? 160 : 161);
            this.mHandler.sendMessage(message);
            Message message2 = new Message();
            message2.what = 3;
            message2.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getSpeed());
            this.mHandler.sendMessage(message2);
            Message message3 = new Message();
            message3.what = 4;
            message3.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getMode());
            this.mHandler.sendMessage(message3);
            int startPosition = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
            Message message4 = new Message();
            message4.what = 7;
            message4.obj = Integer.valueOf(startPosition);
            this.mHandler.sendMessage(message4);
            int stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
            Message message5 = new Message();
            message5.what = 8;
            if (stopPosition <= 0) {
                stopPosition = this.start_stop_position_diff + startPosition;
            }
            message5.obj = Integer.valueOf(stopPosition);
            this.mHandler.sendMessage(message5);
            Message message6 = new Message();
            message6.what = 9;
            message6.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getSmoothness());
            this.mHandler.sendMessage(message6);
            Message message7 = new Message();
            message7.what = 10;
            message7.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getPosition());
            this.mHandler.sendMessage(message7);
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
        this.mModes.clear();
        boolean hasMode = ProductUtil.hasMode(0);
        FrameLayout frameLayout = this.iplay_layout_mode_btn;
        if (hasMode) {
            frameLayout.setVisibility(0);
            this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
        } else {
            frameLayout.setVisibility(8);
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
    }

    private void initRecordPermission() {
        if (ContextCompat.checkSelfPermission(this, "android.permission.RECORD_AUDIO") == 0) {
            startMicClick();
            return;
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.IPlayNewActivity$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                IPlayNewActivity.this.m560x733cdade((Permission) obj);
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
        this.title_txt_status = (TextView) findViewById(R.id.title_txt_status);
        this.iplay_layout = (RelativeLayout) findViewById(R.id.iplay_layout);
        this.iplay_layout_bg = (ImageView) findViewById(R.id.iplay_layout_bg);
        this.img_run_mode = (ImageView) findViewById(R.id.img_run_mode);
        this.iplay_layout_mic_btn = (FrameLayout) findViewById(R.id.iplay_layout_mic_btn);
        this.iplay_img_mic = (ImageView) findViewById(R.id.iplay_img_mic);
        this.iplay_layout_mode_btn = (FrameLayout) findViewById(R.id.iplay_layout_mode_btn);
        this.iplay_img_mode = (ImageView) findViewById(R.id.iplay_img_mode);
        this.txt_speed_label = (TextView) findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) findViewById(R.id.txt_speed);
        this.progress_speed = (ProgressBar) findViewById(R.id.progress_speed);
        this.layout_speed_max = (FrameLayout) findViewById(R.id.layout_speed_max);
        this.txt_speed_max = (TextView) findViewById(R.id.txt_speed_max);
        this.txt_position_label = (TextView) findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) findViewById(R.id.txt_position);
        this.progress_position = (ProgressBar) findViewById(R.id.progress_position);
        this.layout_position_max = (FrameLayout) findViewById(R.id.layout_position_max);
        this.txt_position_max = (TextView) findViewById(R.id.txt_position_max);
        this.layout_position_min = (FrameLayout) findViewById(R.id.layout_position_min);
        this.txt_position_min = (TextView) findViewById(R.id.txt_position_min);
        this.txt_smooth_label = (TextView) findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) findViewById(R.id.txt_smooth);
        this.seek_smooth = (CustomSeekbar) findViewById(R.id.seek_smooth);
        this.iplay_playView = (PlayView_Four) findViewById(R.id.iplay_playView);
        this.iplay_layout_touch_position = (FrameLayout) findViewById(R.id.iplay_layout_touch_position);
        this.iplay_img_position = (ImageView) findViewById(R.id.iplay_img_position);
        this.iplay_layout_mode = (FrameLayout) findViewById(R.id.iplay_layout_mode);
        this.iplay_recycler_mode = (RecyclerView) findViewById(R.id.iplay_recycler_mode);
        this.iplay_layout_arrow_up = (FrameLayout) findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) findViewById(R.id.iplay_layout_arrow_down);
        this.iplay_img_close_mode = (ImageView) findViewById(R.id.iplay_img_close_mode);
        this.iplay_layout_vibration = (FrameLayout) findViewById(R.id.iplay_layout_vibration);
        this.iplay_recycler_vibration = (RecyclerView) findViewById(R.id.iplay_recycler_vibration);
        this.iplay_layout_vibrate_arrow_up = (FrameLayout) findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.iplay_layout_vibrate_arrow_down = (FrameLayout) findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.iplay_img_close_vibration = (ImageView) findViewById(R.id.iplay_img_close_vibration);
        this.img_close_left = (ImageView) findViewById(R.id.img_close_left);
        this.img_close_right = (ImageView) findViewById(R.id.img_close_right);
        this.img_gesture_left = (ImageView) findViewById(R.id.img_gesture_left);
        this.img_gesture_right = (ImageView) findViewById(R.id.img_gesture_right);
        this.txt_gesture_tip = (TextView) findViewById(R.id.txt_gesture_tip);
        this.view_touch = findViewById(R.id.view_touch);
        LanguageUtil.setText(this, this.title_txt_status, "bluetoothDisconnected");
        LanguageUtil.setText(this, this.txt_speed_label, "speedTitle");
        this.txt_position_label.setText(LanguageUtil.getLanguageByKey(this, "position") + ":");
        this.txt_smooth_label.setText(LanguageUtil.getLanguageByKey(this, "smoothness") + ":");
        LanguageUtil.setText(this, this.txt_gesture_tip, "exitDoubleGameNote");
        this.iplay_playView.setProgressBar(this.progress_speed);
        this.iplay_playView.setTouch(false);
        Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.iplay_layout_bg);
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.3
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + IPlayNewActivity.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + IPlayNewActivity.this.seek_smooth.getProgress());
                int progress = IPlayNewActivity.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                int i3 = i / 100;
                IPlayNewActivity.this.txt_smooth.setText(i3 + "");
                BleManager.getInstance(IPlayNewActivity.this).writeSmooth(i3);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.3.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        IPlayNewActivity.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        this.mPositionEnd = i;
        this.txt_position_max.setText(this.mPositionEnd + "");
        double width = this.progress_position.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.layout_position_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_position_max.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionUI(int i) {
        this.mPosition = i;
        this.txt_position.setText(i + "");
        this.progress_position.setProgress(i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSmoothUI(int i) {
        this.txt_smooth.setText(i + "");
        ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.16
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                IPlayNewActivity.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
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
                startAnimation(this.iplay_layout_mode, this.iplay_layout_vibration);
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
        this.mPositionStart = i;
        this.txt_position_min.setText(this.mPositionStart + "");
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.layout_position_min.getLayoutParams();
        layoutParams.setMargins((int) ((this.progress_position.getWidth() / 100.0d) * i), 0, 0, 0);
        this.layout_position_min.setLayoutParams(layoutParams);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void scrollModeHideImg() {
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
        this.img_close_left.setOnTouchListener(this.leftOnTouch);
        this.img_close_right.setOnTouchListener(this.rightOnTouch);
        this.layout_speed_max.setOnTouchListener(this.safeOnTouch);
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
        this.iplay_layout_touch_position.setOnTouchListener(this.positionOnTouch);
        this.iplay_playView.setOnMoveChangeListener(new PlayView_Four.OnMoveChangeListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.5
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnMoveChangeListener
            public void onMoveChange(int i) {
                if (i < 0) {
                    i = 0;
                } else if (i > 100) {
                    i = 100;
                }
                if (i > IPlayNewActivity.this.mSafeMax) {
                    i = IPlayNewActivity.this.mSafeMax;
                }
                if (i != IPlayNewActivity.this.progress_speed.getProgress() || i == 0) {
                    IPlayNewActivity.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (IPlayNewActivity.this.mMode != 0) {
                        IPlayNewActivity.this.mMode = 0;
                        if (IPlayNewActivity.this.mAdapter != null) {
                            IPlayNewActivity.this.mAdapter.setClickPosition(IPlayNewActivity.this.mMode - 1);
                        }
                    }
                    BleManager.getInstance(IPlayNewActivity.this).writeSpeed(i);
                    IPlayNewActivity.this.mSpeed = i;
                }
            }
        });
        this.view_touch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.6
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                IPlayNewActivity.this.hideGestureTip();
                return false;
            }
        });
        BleManager.getInstance(this).setOnConnectStateListener(this);
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.7
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
                IPlayNewActivity.this.mSpeed = i;
                BleDeviceModel bleDeviceModel = BleManager.getInstance(IPlayNewActivity.this).getmConnBleDevice();
                if (bleDeviceModel == null || i != bleDeviceModel.getSpeed()) {
                    BleManager.getInstance(IPlayNewActivity.this).writeSpeed(i);
                }
            }
        });
    }

    private void startAnimation(final FrameLayout frameLayout, final FrameLayout frameLayout2) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 0.0f, 1.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.9
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayNewActivity.this.inAnim = false;
                if (i != IPlayNewActivity.this.isShowMode) {
                    IPlayNewActivity.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayNewActivity.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.10
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayNewActivity.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayNewActivity.this.inAnim = true;
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
        if (!z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic_select);
            this.recorderVolumeManager.startRecording();
        } else {
            imageView.setImageResource(R.mipmap.ic_iplay_mic);
            this.recorderVolumeManager.stopRecording();
            BleManager.getInstance(this).writeSpeed(0);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.11
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                IPlayNewActivity.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != IPlayNewActivity.this.isShowMode) {
                    IPlayNewActivity.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                IPlayNewActivity.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    private void unregisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-IPlayNewActivity, reason: not valid java name */
    /* synthetic */ void m560x733cdade(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            startMicClick();
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            DialogUtil.showConfirmDialog(this, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.IPlayNewActivity.8
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + IPlayNewActivity.this.getPackageName()));
                    IPlayNewActivity.this.startActivity(intent);
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
        if (view.getId() == R.id.img_run_mode) {
            if (this.runMode) {
                Message message = new Message();
                message.what = 1;
                message.obj = 161;
                this.mHandler.sendMessage(message);
                BleManager.getInstance(this).writeRunManual();
                return;
            }
            Message message2 = new Message();
            message2.what = 1;
            message2.obj = 160;
            this.mHandler.sendMessage(message2);
            BleManager.getInstance(this).writeRunAuto();
            return;
        }
        if (view.getId() != R.id.iplay_layout_mode_btn) {
            if (view.getId() == R.id.iplay_layout_mic_btn) {
                initRecordPermission();
            }
        } else {
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
            } else {
                startAnimation(frameLayout, null);
            }
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(int i) {
        if (i > 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayNewActivity.19
                @Override // java.lang.Runnable
                public void run() {
                    IPlayNewActivity.this.title_txt_status.setVisibility(8);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.IPlayNewActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
            }
        });
        setContentView(R.layout.activity_iplay_new);
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
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.IPlayNewActivity.20
                @Override // java.lang.Runnable
                public void run() {
                    IPlayNewActivity.this.title_txt_status.setVisibility(0);
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
        if (f4 <= -9.0f && f4 > -10.0f) {
            int i = this.reversalCount + 1;
            this.reversalCount = i;
            if (i >= 4) {
                this.recoverCount = 0;
                if (!this.isReversal && this.runMode) {
                    LogUtil.e("Sensor value 翻转翻转翻转");
                    this.isReversal = true;
                    BleManager.getInstance(this).writeSpeed(0);
                }
            }
        } else if (f4 > -1.0f) {
            int i2 = this.recoverCount + 1;
            this.recoverCount = i2;
            if (i2 >= 4) {
                this.reversalCount = 0;
                if (this.isReversal && this.runMode) {
                    LogUtil.e("Sensor value 翻转恢复");
                    this.isReversal = false;
                    if (this.mMode <= 0) {
                        BleManager.getInstance(this).writeSpeed(this.progress_speed.getProgress());
                    } else if (this.mModes.size() >= this.mMode) {
                        BleManager.getInstance(this).writeData(this.mModes.get(this.mMode - 1).getmCommand());
                    } else {
                        BleManager.getInstance(this).writeSpeedMode(this.mMode);
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
