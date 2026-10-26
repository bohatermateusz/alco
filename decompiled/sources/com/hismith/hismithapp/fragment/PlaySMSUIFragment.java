package com.hismith.hismithapp.fragment;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.view.animation.LinearInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.PlayView_Four;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMSUIFragment extends SocketPlayFragment implements View.OnClickListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private BaseActivity baseActivity;
    private float downX;
    private ImageView img_close_left;
    private ImageView img_close_right;
    private ImageView img_run_mode;
    private ImageView iplay_img_close_mode;
    private ImageView iplay_img_close_vibration;
    private ImageView iplay_img_mic;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_position;
    private RelativeLayout iplay_layout;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
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
    private Gson mGson;
    private GridLayoutManager mLayoutManager;
    private List<ProductModel.Mode> mModes;
    private int mSafeMax;
    private ProgressBar progress_position;
    private ProgressBar progress_speed;
    private CustomSeekbar seek_smooth;
    private TextView txt_position;
    private TextView txt_position_label;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private TextView txt_smooth;
    private TextView txt_smooth_label;
    private TextView txt_speed;
    private TextView txt_speed_max;
    private int leftAction = 1;
    private int rightAction = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        /* JADX WARN: Code restructure failed: missing block: B:24:0x0087, code lost:
        
            return false;
         */
        @Override // android.os.Handler.Callback
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public boolean handleMessage(Message message) {
            int intValue = ((Integer) message.obj).intValue();
            Bundle data = message.getData();
            String string = data != null ? data.getString(Constans.BleType) : "";
            switch (message.what) {
                case 1:
                    if (intValue != 160) {
                        if (intValue == 161) {
                            PlaySMSUIFragment.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_off);
                            break;
                        }
                    } else {
                        PlaySMSUIFragment.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_on);
                        break;
                    }
                    break;
                case 2:
                    if (intValue == 0) {
                        PlaySMSUIFragment.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_on);
                        PlaySMSUIFragment.this.refreshSpeedUI(0);
                        PlaySMSUIFragment.this.refreshPositionUI(0);
                        break;
                    }
                    break;
                case 3:
                    PlaySMSUIFragment.this.refreshSpeedUI(intValue);
                    break;
                case 4:
                    PlaySMSUIFragment.this.refreshSpeedModeUI(intValue);
                    break;
                case 7:
                    if (!"CC".equals(string)) {
                        PlaySMSUIFragment.this.refreshStartBitUI(intValue);
                        break;
                    }
                    break;
                case 8:
                    if (!"CC".equals(string)) {
                        PlaySMSUIFragment.this.refreshEndBitUI(intValue);
                        break;
                    }
                    break;
                case 9:
                    PlaySMSUIFragment.this.refreshSmoothUI(intValue);
                    break;
                case 10:
                    PlaySMSUIFragment.this.refreshPositionUI(intValue);
                    break;
            }
        }
    });
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.4
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMSUIFragment.this.downX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlaySMSUIFragment.this.downX;
            int right = PlaySMSUIFragment.this.progress_speed.getRight() + (PlaySMSUIFragment.this.layout_speed_max.getWidth() / 2);
            LogUtil.e("getRight():" + PlaySMSUIFragment.this.layout_speed_max.getRight() + ", width:" + PlaySMSUIFragment.this.progress_speed.getRight() + ", " + PlaySMSUIFragment.this.layout_speed_max.getWidth());
            int right2 = (int) (right - (PlaySMSUIFragment.this.layout_speed_max.getRight() + x));
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > PlaySMSUIFragment.this.progress_speed.getWidth()) {
                right2 = PlaySMSUIFragment.this.progress_speed.getWidth();
            }
            LogUtil.e("safe right:" + right2);
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlaySMSUIFragment.this.layout_speed_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            PlaySMSUIFragment.this.layout_speed_max.setLayoutParams(layoutParams);
            int width = (int) (100.0d - ((right2 * 100.0d) / PlaySMSUIFragment.this.progress_speed.getWidth()));
            PlaySMSUIFragment.this.mSafeMax = width;
            PlaySMSUIFragment.this.txt_speed_max.setText(width + "");
            PlaySMSUIFragment playSMSUIFragment = PlaySMSUIFragment.this;
            playSMSUIFragment.onSetSafeMax(playSMSUIFragment.mSafeMax);
            if (width >= 0 && width < PlaySMSUIFragment.this.progress_speed.getSecondaryProgress()) {
                BleManager.getInstance(PlaySMSUIFragment.this.baseActivity).writeSpeed(width);
                PlaySMSUIFragment.this.progress_speed.setProgress(width);
                PlaySMSUIFragment.this.txt_speed.setText(width + "");
                return true;
            }
            if (width < PlaySMSUIFragment.this.progress_speed.getSecondaryProgress() || PlaySMSUIFragment.this.progress_speed.getSecondaryProgress() == PlaySMSUIFragment.this.progress_speed.getProgress()) {
                return true;
            }
            BleManager.getInstance(PlaySMSUIFragment.this.baseActivity).writeSpeed(PlaySMSUIFragment.this.progress_speed.getSecondaryProgress());
            PlaySMSUIFragment.this.progress_speed.setProgress(PlaySMSUIFragment.this.progress_speed.getSecondaryProgress());
            PlaySMSUIFragment.this.txt_speed.setText(PlaySMSUIFragment.this.progress_speed.getSecondaryProgress() + "");
            return true;
        }
    };
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.5
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMSUIFragment.this.leftAction = 0;
                PlaySMSUIFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMSUIFragment.this.lastLeftClickTime;
                PlaySMSUIFragment playSMSUIFragment = PlaySMSUIFragment.this;
                if (j < 500) {
                    boolean z = playSMSUIFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMSUIFragment playSMSUIFragment2 = PlaySMSUIFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMSUIFragment2.baseActivity).exit();
                    } else if (playSMSUIFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMSUIFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMSUIFragment.lastLeftClickTime = currentTimeMillis;
                PlaySMSUIFragment.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.6
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMSUIFragment.this.rightAction = 0;
                PlaySMSUIFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMSUIFragment.this.lastRightClickTime;
                PlaySMSUIFragment playSMSUIFragment = PlaySMSUIFragment.this;
                if (j < 500) {
                    boolean z = playSMSUIFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMSUIFragment playSMSUIFragment2 = PlaySMSUIFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMSUIFragment2.baseActivity).exit();
                    } else if (playSMSUIFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMSUIFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMSUIFragment.lastRightClickTime = currentTimeMillis;
                PlaySMSUIFragment.this.rightAction = 1;
            }
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.8
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                PlaySMSUIFragment.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    PlaySMSUIFragment.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;

    public PlaySMSUIFragment(int i) {
        this.mSafeMax = 100;
        this.mSafeMax = i;
    }

    private void closeSocket() {
        if (SocketManager.getInstance().getWebSocket() != null) {
            LogUtil.e("WebSocket 发送：关闭");
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exitIPlay() {
        if (this.leftAction == 0 && this.rightAction == 0) {
            BaseActivity baseActivity = this.baseActivity;
            if (baseActivity instanceof ControlSponsorActivity2) {
                ((ControlSponsorActivity2) baseActivity).exit();
            } else if (baseActivity instanceof ControlAcceptActivity2) {
                ((ControlAcceptActivity2) baseActivity).exit();
            }
        }
        if (this.leftAction == 1) {
            LogUtil.e("onTouch 左抖动");
            shakeAnimation(this.img_close_left);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            LogUtil.e("onTouch 右抖动");
            shakeAnimation(this.img_close_right);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
    }

    private void init() {
        this.mGson = new Gson();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.iplay_recycler_mode.setLayoutManager(gridLayoutManager);
        this.mModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = iplayModeAdapter;
        this.iplay_recycler_mode.setAdapter(iplayModeAdapter);
        initProductData();
        initBleDataUI();
        registerReceiver();
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
            Message message4 = new Message();
            message4.what = 7;
            message4.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStartPosition());
            this.mHandler.sendMessage(message4);
            Message message5 = new Message();
            message5.what = 8;
            message5.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStopPosition());
            this.mHandler.sendMessage(message5);
            Message message6 = new Message();
            message6.what = 9;
            message6.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSmoothness());
            this.mHandler.sendMessage(message6);
            this.txt_position.setText(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getPosition() + "");
            this.progress_position.setProgress(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getPosition());
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

    private void initView(View view) {
        this.iplay_layout = (RelativeLayout) view.findViewById(R.id.iplay_layout);
        this.img_run_mode = (ImageView) view.findViewById(R.id.img_run_mode);
        this.iplay_layout_mic_btn = (FrameLayout) view.findViewById(R.id.iplay_layout_mic_btn);
        this.iplay_img_mic = (ImageView) view.findViewById(R.id.iplay_img_mic);
        this.iplay_layout_mode_btn = (FrameLayout) view.findViewById(R.id.iplay_layout_mode_btn);
        this.iplay_img_mode = (ImageView) view.findViewById(R.id.iplay_img_mode);
        TextView textView = (TextView) view.findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) view.findViewById(R.id.txt_speed);
        this.progress_speed = (ProgressBar) view.findViewById(R.id.progress_speed);
        this.layout_speed_max = (FrameLayout) view.findViewById(R.id.layout_speed_max);
        this.txt_speed_max = (TextView) view.findViewById(R.id.txt_speed_max);
        this.txt_position_label = (TextView) view.findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) view.findViewById(R.id.txt_position);
        this.progress_position = (ProgressBar) view.findViewById(R.id.progress_position);
        this.layout_position_max = (FrameLayout) view.findViewById(R.id.layout_position_max);
        this.txt_position_max = (TextView) view.findViewById(R.id.txt_position_max);
        this.layout_position_min = (FrameLayout) view.findViewById(R.id.layout_position_min);
        this.txt_position_min = (TextView) view.findViewById(R.id.txt_position_min);
        this.txt_smooth_label = (TextView) view.findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) view.findViewById(R.id.txt_smooth);
        this.seek_smooth = (CustomSeekbar) view.findViewById(R.id.seek_smooth);
        this.iplay_playView = (PlayView_Four) view.findViewById(R.id.iplay_playView);
        this.iplay_layout_touch_position = (FrameLayout) view.findViewById(R.id.iplay_layout_touch_position);
        this.iplay_img_position = (ImageView) view.findViewById(R.id.iplay_img_position);
        this.iplay_layout_mode = (FrameLayout) view.findViewById(R.id.iplay_layout_mode);
        this.iplay_recycler_mode = (RecyclerView) view.findViewById(R.id.iplay_recycler_mode);
        this.iplay_layout_arrow_up = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_down);
        this.iplay_img_close_mode = (ImageView) view.findViewById(R.id.iplay_img_close_mode);
        this.iplay_layout_vibration = (FrameLayout) view.findViewById(R.id.iplay_layout_vibration);
        this.iplay_recycler_vibration = (RecyclerView) view.findViewById(R.id.iplay_recycler_vibration);
        this.iplay_layout_vibrate_arrow_up = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.iplay_layout_vibrate_arrow_down = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.iplay_img_close_vibration = (ImageView) view.findViewById(R.id.iplay_img_close_vibration);
        this.img_close_left = (ImageView) view.findViewById(R.id.img_close_left);
        this.img_close_right = (ImageView) view.findViewById(R.id.img_close_right);
        LanguageUtil.setText(this.baseActivity, textView, "speedTitle");
        this.txt_position_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "position") + ":");
        this.txt_smooth_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "smoothness") + ":");
        this.iplay_playView.setProgressBar(this.progress_speed);
        this.iplay_playView.setTouch(false);
        this.seek_smooth.setTouch(false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        this.txt_position_max.setText(i + "");
        double width = this.progress_position.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.layout_position_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_position_max.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionUI(int i) {
        this.txt_position.setText(i + "");
        this.progress_position.setProgress(i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.mSafeMax = i;
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
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.7
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                PlaySMSUIFragment.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
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
        this.txt_position_min.setText(i + "");
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.layout_position_min.getLayoutParams();
        layoutParams.setMargins((int) ((this.progress_position.getWidth() / 100.0d) * i), 0, 0, 0);
        this.layout_position_min.setLayoutParams(layoutParams);
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
        this.iplay_layout_mode_btn.setOnClickListener(this);
        this.img_close_left.setOnTouchListener(this.leftOnTouch);
        this.img_close_right.setOnTouchListener(this.rightOnTouch);
        this.layout_speed_max.setOnTouchListener(this.safeOnTouch);
        this.layout_speed_max.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.2
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlaySMSUIFragment.this.layout_speed_max.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                PlaySMSUIFragment playSMSUIFragment = PlaySMSUIFragment.this;
                playSMSUIFragment.refreshSafeUI(playSMSUIFragment.mSafeMax);
            }
        });
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.3
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMSUIFragment.this.scrollModeHideImg();
            }
        });
    }

    private void setSmooth(int i) {
        Message message = new Message();
        message.what = 9;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
    }

    private void setSpeed(int i) {
        Message message = new Message();
        message.what = 3;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
    }

    private void setSpeedMode(int i) {
        Message message = new Message();
        message.what = 4;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
    }

    private void setStrokeMax(int i) {
        Message message = new Message();
        message.what = 8;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
    }

    private void setStrokeMin(int i) {
        Message message = new Message();
        message.what = 7;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
    }

    private void setStrokeNow(int i) {
    }

    private void setStrokeType(int i) {
        Message message = new Message();
        message.what = 1;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.9
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMSUIFragment.this.inAnim = false;
                if (i != PlaySMSUIFragment.this.isShowMode) {
                    PlaySMSUIFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMSUIFragment.this.inAnim = true;
                frameLayout.setVisibility(0);
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.10
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMSUIFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != PlaySMSUIFragment.this.isShowMode) {
                    PlaySMSUIFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMSUIFragment.this.inAnim = true;
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

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
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

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initProductData() {
        if (MyApplication.getApplication().getmProductModel() == null) {
            return;
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

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment, com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() != R.id.iplay_layout_mode_btn || this.inAnim) {
            return;
        }
        if (this.iplay_layout_mode.getVisibility() == 0) {
            stopAnimation(this.iplay_layout_mode);
        } else {
            this.isShowMode = 1;
            startAnimation(this.iplay_layout_mode);
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_iplay_new_ui, viewGroup, false);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        unregisterReceiver();
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onNotify(String str) {
        try {
            LogUtil.e("PlaySMFragment onNotify:" + str);
            SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel != null) {
                String action = socketSpeedModel.getAction();
                if ("setSpeed".equals(action)) {
                    if (!socketSpeedModel.isMotor1()) {
                        socketSpeedModel.isMotor2();
                        return;
                    }
                    final int speed = socketSpeedModel.getSpeed();
                    if (speed > 100) {
                        speed = 100;
                    } else if (speed < 0) {
                        speed = 0;
                    }
                    this.baseActivity.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.fragment.PlaySMSUIFragment.11
                        @Override // java.lang.Runnable
                        public void run() {
                            PlaySMSUIFragment.this.progress_speed.setSecondaryProgress(speed);
                        }
                    });
                    int i = this.mSafeMax;
                    if (speed > i) {
                        speed = i;
                    }
                    setSpeed(speed);
                    return;
                }
                if ("setMode".equals(action)) {
                    int mode = socketSpeedModel.getMode();
                    if (socketSpeedModel.isMotor1Mode()) {
                        setSpeedMode(mode);
                        return;
                    } else {
                        socketSpeedModel.isMotor2Mode();
                        return;
                    }
                }
                if ("setStroke".equals(action)) {
                    setStrokeMin(socketSpeedModel.getStrokeMin());
                    setStrokeMax(socketSpeedModel.getStrokeMax());
                } else if ("setSmooth".equals(action)) {
                    setSmooth(socketSpeedModel.getSmooth());
                } else if ("setStrokeType".equals(action)) {
                    setStrokeType(socketSpeedModel.getStrokeType());
                } else if ("setStrokeNow".equals(action)) {
                    setStrokeNow(socketSpeedModel.getStrokeNow());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void shakeAnimation(View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(this.baseActivity, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }
}
