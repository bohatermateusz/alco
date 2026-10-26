package com.hismith.hismithapp.fragment;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Point;
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
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMVUIFragment extends SocketPlayFragment implements View.OnClickListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private BaseActivity baseActivity;
    private float downY;
    private ImageView iplay_img_close_left;
    private ImageView iplay_img_close_mode;
    private ImageView iplay_img_close_right;
    private ImageView iplay_img_close_vibration;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_music;
    private ImageView iplay_img_vibrate;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
    private FrameLayout iplay_layout_mode;
    private LinearLayout iplay_layout_mode_border;
    private ViewGroup iplay_layout_speed_safe;
    private FrameLayout iplay_layout_vibrate_arrow_down;
    private FrameLayout iplay_layout_vibrate_arrow_up;
    private ViewGroup iplay_layout_vibrate_safe;
    private FrameLayout iplay_layout_vibration;
    private LinearLayout iplay_layout_vibration_border;
    private PlayView_Four iplay_playView;
    private PlayView_Four iplay_playView_vibrate;
    private VerticalProgressBar iplay_progress_speed;
    private VerticalProgressBar iplay_progress_vibrate;
    private RecyclerView iplay_recycler_mode;
    private RecyclerView iplay_recycler_vibration;
    private TextView iplay_txt_max_speed;
    private TextView iplay_txt_max_vibrate;
    private TextView iplay_txt_min_speed;
    private TextView iplay_txt_min_vibrate;
    private TextView iplay_txt_speed;
    private TextView iplay_txt_speed_safe;
    private TextView iplay_txt_vibrate;
    private TextView iplay_txt_vibrate_safe;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private Gson mGson;
    private GridLayoutManager mLayoutManager;
    private List<ProductModel.Mode> mModes;
    private int mSafeMax;
    private int mVibrateSafeMax;
    private IplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private List<ProductModel.Mode> mVibrationModes;
    private float vibrateDownY;
    private int mode = -1;
    private int vibrationMode = -1;
    private int music = 0;
    private int leftAction = 1;
    private int rightAction = 1;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_VIBRATION_MODE = 5;
    private final int WHAT_MUSIC = 6;
    private final int WHAT_VIBRATE = 7;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                int intValue = ((Integer) message.obj).intValue();
                PlaySMVUIFragment.this.iplay_txt_speed.setText(intValue + "");
                PlaySMVUIFragment.this.iplay_progress_speed.setProgress(intValue);
                PlaySMVUIFragment.this.isShowMode = 0;
                if (PlaySMVUIFragment.this.iplay_layout_mode.getVisibility() == 0) {
                    PlaySMVUIFragment playSMVUIFragment = PlaySMVUIFragment.this;
                    playSMVUIFragment.stopAnimation(playSMVUIFragment.iplay_layout_mode);
                }
                PlaySMVUIFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                PlaySMVUIFragment.this.mode = 0;
                if (PlaySMVUIFragment.this.mAdapter != null) {
                    PlaySMVUIFragment.this.mAdapter.setClickPosition(-1);
                }
            } else if (i == 2) {
                PlaySMVUIFragment.this.mode = ((Integer) message.obj).intValue();
                int i2 = PlaySMVUIFragment.this.mode;
                PlaySMVUIFragment playSMVUIFragment2 = PlaySMVUIFragment.this;
                if (i2 <= 0) {
                    playSMVUIFragment2.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                } else {
                    playSMVUIFragment2.isShowMode = 1;
                    if (PlaySMVUIFragment.this.iplay_layout_mode.getVisibility() != 0) {
                        PlaySMVUIFragment playSMVUIFragment3 = PlaySMVUIFragment.this;
                        playSMVUIFragment3.startAnimation(playSMVUIFragment3.iplay_layout_mode, PlaySMVUIFragment.this.iplay_layout_vibration);
                    }
                    PlaySMVUIFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                }
                if (PlaySMVUIFragment.this.mAdapter != null) {
                    PlaySMVUIFragment.this.mAdapter.setClickPosition(PlaySMVUIFragment.this.mode - 1);
                    if (PlaySMVUIFragment.this.mode > 0) {
                        PlaySMVUIFragment.this.iplay_recycler_mode.smoothScrollToPosition(PlaySMVUIFragment.this.mode - 1);
                    }
                }
            } else if (i == 5) {
                PlaySMVUIFragment.this.vibrationMode = ((Integer) message.obj).intValue();
                int i3 = PlaySMVUIFragment.this.vibrationMode;
                PlaySMVUIFragment playSMVUIFragment4 = PlaySMVUIFragment.this;
                if (i3 >= 1) {
                    playSMVUIFragment4.isShowMode = 2;
                    if (PlaySMVUIFragment.this.iplay_layout_vibration.getVisibility() != 0) {
                        PlaySMVUIFragment playSMVUIFragment5 = PlaySMVUIFragment.this;
                        playSMVUIFragment5.startAnimation(playSMVUIFragment5.iplay_layout_vibration, PlaySMVUIFragment.this.iplay_layout_mode);
                    }
                    PlaySMVUIFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                } else {
                    playSMVUIFragment4.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                }
                if (PlaySMVUIFragment.this.mVibrationAdapter != null) {
                    PlaySMVUIFragment.this.mVibrationAdapter.setClickPosition(PlaySMVUIFragment.this.vibrationMode - 1);
                    if (PlaySMVUIFragment.this.vibrationMode != 0) {
                        PlaySMVUIFragment.this.iplay_recycler_vibration.smoothScrollToPosition(PlaySMVUIFragment.this.vibrationMode);
                    }
                }
            } else if (i == 6) {
                PlaySMVUIFragment.this.music = ((Integer) message.obj).intValue();
                int i4 = PlaySMVUIFragment.this.music;
                PlaySMVUIFragment playSMVUIFragment6 = PlaySMVUIFragment.this;
                if (i4 > 0) {
                    playSMVUIFragment6.iplay_img_music.setImageResource(R.mipmap.ic_iplay_voice_select);
                } else {
                    playSMVUIFragment6.iplay_img_music.setImageResource(R.mipmap.ic_iplay_voice);
                }
            } else if (i == 7) {
                int intValue2 = ((Integer) message.obj).intValue();
                PlaySMVUIFragment.this.iplay_txt_vibrate.setText(intValue2 + "");
                PlaySMVUIFragment.this.iplay_progress_vibrate.setProgress(intValue2);
                PlaySMVUIFragment.this.isShowMode = 0;
                if (PlaySMVUIFragment.this.iplay_layout_vibration.getVisibility() == 0) {
                    PlaySMVUIFragment playSMVUIFragment7 = PlaySMVUIFragment.this;
                    playSMVUIFragment7.stopAnimation(playSMVUIFragment7.iplay_layout_vibration);
                }
                PlaySMVUIFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                PlaySMVUIFragment.this.vibrationMode = 0;
                if (PlaySMVUIFragment.this.mVibrationAdapter != null) {
                    PlaySMVUIFragment.this.mVibrationAdapter.setClickPosition(-1);
                }
            }
            return false;
        }
    });
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.9
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMVUIFragment.this.leftAction = 0;
                PlaySMVUIFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMVUIFragment.this.lastLeftClickTime;
                PlaySMVUIFragment playSMVUIFragment = PlaySMVUIFragment.this;
                if (j < 500) {
                    boolean z = playSMVUIFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMVUIFragment playSMVUIFragment2 = PlaySMVUIFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMVUIFragment2.baseActivity).exit();
                    } else if (playSMVUIFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMVUIFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMVUIFragment.lastLeftClickTime = currentTimeMillis;
                PlaySMVUIFragment.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.10
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMVUIFragment.this.rightAction = 0;
                PlaySMVUIFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMVUIFragment.this.lastRightClickTime;
                PlaySMVUIFragment playSMVUIFragment = PlaySMVUIFragment.this;
                if (j < 500) {
                    boolean z = playSMVUIFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMVUIFragment playSMVUIFragment2 = PlaySMVUIFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMVUIFragment2.baseActivity).exit();
                    } else if (playSMVUIFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMVUIFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMVUIFragment.lastRightClickTime = currentTimeMillis;
                PlaySMVUIFragment.this.rightAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener speedSafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.11
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMVUIFragment.this.downY = motionEvent.getY();
                LogUtil.e("onTouch Down:" + PlaySMVUIFragment.this.downY);
                return true;
            }
            if (action != 2) {
                return true;
            }
            int top = (int) (view.getTop() + (motionEvent.getY() - PlaySMVUIFragment.this.downY));
            if (top < 0) {
                top = 0;
            } else if (top > PlaySMVUIFragment.this.iplay_progress_speed.getHeight()) {
                top = PlaySMVUIFragment.this.iplay_progress_speed.getHeight();
            }
            ViewGroup.LayoutParams layoutParams = PlaySMVUIFragment.this.iplay_layout_speed_safe.getLayoutParams();
            if (PlaySMVUIFragment.this.iplay_layout_speed_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
                ((LinearLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            } else if (PlaySMVUIFragment.this.iplay_layout_speed_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
                ((FrameLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            }
            PlaySMVUIFragment.this.iplay_layout_speed_safe.setLayoutParams(layoutParams);
            int height = 100 - (((int) (top * 100.0d)) / PlaySMVUIFragment.this.iplay_progress_speed.getHeight());
            PlaySMVUIFragment.this.iplay_txt_speed_safe.setText("" + height);
            PlaySMVUIFragment.this.mSafeMax = height;
            PlaySMVUIFragment playSMVUIFragment = PlaySMVUIFragment.this;
            playSMVUIFragment.onSetSafeMax(playSMVUIFragment.mSafeMax);
            if (height >= 0 && height < PlaySMVUIFragment.this.iplay_progress_speed.getSecondaryProgress()) {
                BleManager.getInstance(PlaySMVUIFragment.this.baseActivity).writeSpeed(height);
                return true;
            }
            if (height < PlaySMVUIFragment.this.iplay_progress_speed.getSecondaryProgress() || PlaySMVUIFragment.this.iplay_progress_speed.getSecondaryProgress() == PlaySMVUIFragment.this.iplay_progress_speed.getProgress()) {
                return true;
            }
            BleManager.getInstance(PlaySMVUIFragment.this.baseActivity).writeSpeed(PlaySMVUIFragment.this.iplay_progress_speed.getSecondaryProgress());
            return true;
        }
    };
    private View.OnTouchListener vibrateSafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.12
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMVUIFragment.this.vibrateDownY = motionEvent.getY();
                return true;
            }
            if (action != 2) {
                return true;
            }
            int top = (int) (view.getTop() + (motionEvent.getY() - PlaySMVUIFragment.this.vibrateDownY));
            if (top < 0) {
                top = 0;
            } else if (top > PlaySMVUIFragment.this.iplay_progress_vibrate.getHeight()) {
                top = PlaySMVUIFragment.this.iplay_progress_vibrate.getHeight();
            }
            ViewGroup.LayoutParams layoutParams = PlaySMVUIFragment.this.iplay_layout_vibrate_safe.getLayoutParams();
            if (PlaySMVUIFragment.this.iplay_layout_vibrate_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
                ((LinearLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            } else if (PlaySMVUIFragment.this.iplay_layout_vibrate_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
                ((FrameLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            }
            PlaySMVUIFragment.this.iplay_layout_vibrate_safe.setLayoutParams(layoutParams);
            int height = 100 - (((int) (top * 100.0d)) / PlaySMVUIFragment.this.iplay_progress_vibrate.getHeight());
            PlaySMVUIFragment.this.iplay_txt_vibrate_safe.setText("" + height);
            PlaySMVUIFragment.this.mVibrateSafeMax = height;
            PlaySMVUIFragment playSMVUIFragment = PlaySMVUIFragment.this;
            playSMVUIFragment.onSetVibrateSafeMax(playSMVUIFragment.mVibrateSafeMax);
            if (height >= 0 && height < PlaySMVUIFragment.this.iplay_progress_vibrate.getSecondaryProgress()) {
                BleManager.getInstance(PlaySMVUIFragment.this.baseActivity).writeVibrate(height);
                return true;
            }
            if (height < PlaySMVUIFragment.this.iplay_progress_vibrate.getSecondaryProgress() || PlaySMVUIFragment.this.iplay_progress_vibrate.getSecondaryProgress() == PlaySMVUIFragment.this.iplay_progress_vibrate.getProgress()) {
                return true;
            }
            BleManager.getInstance(PlaySMVUIFragment.this.baseActivity).writeVibrate(PlaySMVUIFragment.this.iplay_progress_vibrate.getSecondaryProgress());
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.13
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                PlaySMVUIFragment.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("AA".equals(stringExtra)) {
                    PlaySMVUIFragment.this.initBleAAData(stringExtra2, stringExtra3);
                } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    PlaySMVUIFragment.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };

    public PlaySMVUIFragment(int i, int i2) {
        this.mSafeMax = 100;
        this.mVibrateSafeMax = 100;
        this.mSafeMax = i;
        this.mVibrateSafeMax = i2;
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
            shakeAnimation(this.iplay_img_close_left);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            LogUtil.e("onTouch 右抖动");
            shakeAnimation(this.iplay_img_close_right);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
    }

    private void init() {
        this.mGson = new Gson();
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
        this.iplay_recycler_mode.setAdapter(iplayModeAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        this.iplay_recycler_vibration.setAdapter(iplayModeAdapter2);
        initProductData();
        BleDeviceModel bleDeviceModel = BleManager.getInstance(this.baseActivity).getmConnBleDevice();
        if (bleDeviceModel != null) {
            this.iplay_txt_speed.setText(bleDeviceModel.getSpeed() + "");
            this.iplay_progress_speed.setProgress(bleDeviceModel.getSpeed());
            int mode = bleDeviceModel.getMode();
            ImageView imageView = this.iplay_img_mode;
            if (mode > 0) {
                imageView.setImageResource(R.mipmap.ic_iplay_mode_select);
                IplayModeAdapter iplayModeAdapter3 = this.mAdapter;
                if (iplayModeAdapter3 != null) {
                    iplayModeAdapter3.setClickPosition(bleDeviceModel.getMode() - 1);
                }
            } else {
                imageView.setImageResource(R.mipmap.ic_iplay_mode);
            }
            int vibrationMode = bleDeviceModel.getVibrationMode();
            ImageView imageView2 = this.iplay_img_vibrate;
            if (vibrationMode > 0) {
                imageView2.setImageResource(R.mipmap.ic_iplay_vibration_select);
                IplayModeAdapter iplayModeAdapter4 = this.mVibrationAdapter;
                if (iplayModeAdapter4 != null) {
                    iplayModeAdapter4.setClickPosition(bleDeviceModel.getVibrationMode() - 1);
                }
            } else {
                imageView2.setImageResource(R.mipmap.ic_iplay_vibration);
            }
            int music = bleDeviceModel.getMusic();
            this.music = music;
            ImageView imageView3 = this.iplay_img_music;
            if (music > 0) {
                imageView3.setImageResource(R.mipmap.ic_iplay_voice_select);
            } else {
                imageView3.setImageResource(R.mipmap.ic_iplay_voice);
            }
        }
        scrollModeHideImg();
        scrollVibrateHideImg();
        registerReceiver();
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

    private void initView(View view) {
        this.iplay_playView = (PlayView_Four) view.findViewById(R.id.iplay_playView);
        this.iplay_playView_vibrate = (PlayView_Four) view.findViewById(R.id.iplay_playView_vibrate);
        this.iplay_progress_speed = (VerticalProgressBar) view.findViewById(R.id.iplay_progress_speed);
        this.iplay_txt_max_speed = (TextView) view.findViewById(R.id.iplay_txt_max_speed);
        this.iplay_txt_min_speed = (TextView) view.findViewById(R.id.iplay_txt_min_speed);
        this.iplay_txt_speed = (TextView) view.findViewById(R.id.iplay_txt_speed);
        this.iplay_progress_vibrate = (VerticalProgressBar) view.findViewById(R.id.iplay_progress_vibrate);
        this.iplay_txt_max_vibrate = (TextView) view.findViewById(R.id.iplay_txt_max_vibrate);
        this.iplay_txt_min_vibrate = (TextView) view.findViewById(R.id.iplay_txt_min_vibrate);
        this.iplay_txt_vibrate = (TextView) view.findViewById(R.id.iplay_txt_vibrate);
        this.iplay_layout_mode = (FrameLayout) view.findViewById(R.id.iplay_layout_mode);
        this.iplay_layout_mode_border = (LinearLayout) view.findViewById(R.id.iplay_layout_mode_border);
        this.iplay_recycler_mode = (RecyclerView) view.findViewById(R.id.iplay_recycler_mode);
        this.iplay_layout_arrow_up = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_down);
        this.iplay_img_close_mode = (ImageView) view.findViewById(R.id.iplay_img_close_mode);
        this.iplay_layout_vibration = (FrameLayout) view.findViewById(R.id.iplay_layout_vibration);
        this.iplay_layout_vibration_border = (LinearLayout) view.findViewById(R.id.iplay_layout_vibration_border);
        this.iplay_recycler_vibration = (RecyclerView) view.findViewById(R.id.iplay_recycler_vibration);
        this.iplay_layout_vibrate_arrow_up = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.iplay_layout_vibrate_arrow_down = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.iplay_img_close_vibration = (ImageView) view.findViewById(R.id.iplay_img_close_vibration);
        this.iplay_img_mode = (ImageView) view.findViewById(R.id.iplay_img_mode);
        this.iplay_img_music = (ImageView) view.findViewById(R.id.iplay_img_music);
        this.iplay_img_vibrate = (ImageView) view.findViewById(R.id.iplay_img_vibrate);
        this.iplay_img_close_left = (ImageView) view.findViewById(R.id.iplay_img_close_left);
        this.iplay_img_close_right = (ImageView) view.findViewById(R.id.iplay_img_close_right);
        this.iplay_layout_speed_safe = (ViewGroup) view.findViewById(R.id.iplay_layout_speed_safe);
        this.iplay_layout_vibrate_safe = (ViewGroup) view.findViewById(R.id.iplay_layout_vibrate_safe);
        this.iplay_txt_speed_safe = (TextView) view.findViewById(R.id.iplay_txt_speed_safe);
        this.iplay_txt_vibrate_safe = (TextView) view.findViewById(R.id.iplay_txt_vibrate_safe);
        this.iplay_playView.setProgressBar(this.iplay_progress_speed);
        this.iplay_playView.setTouch(false);
        this.iplay_playView_vibrate.setProgressBar(this.iplay_progress_vibrate);
        this.iplay_playView_vibrate.setTouch(false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.iplay_txt_speed_safe.setText("" + i);
        double height = this.iplay_progress_speed.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        ViewGroup.LayoutParams layoutParams = this.iplay_layout_speed_safe.getLayoutParams();
        if (this.iplay_layout_speed_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
            ((LinearLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        } else if (this.iplay_layout_speed_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
            ((FrameLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        }
        this.iplay_layout_speed_safe.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshVibrateSafeUI(int i) {
        this.iplay_txt_vibrate_safe.setText("" + i);
        double height = this.iplay_progress_vibrate.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        ViewGroup.LayoutParams layoutParams = this.iplay_layout_vibrate_safe.getLayoutParams();
        if (this.iplay_layout_vibrate_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
            ((LinearLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        } else if (this.iplay_layout_vibrate_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
            ((FrameLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        }
        this.iplay_layout_vibrate_safe.setLayoutParams(layoutParams);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        boolean canScrollVertically = this.iplay_recycler_mode.canScrollVertically(1);
        FrameLayout frameLayout = this.iplay_layout_arrow_down;
        if (!canScrollVertically) {
            frameLayout.setVisibility(4);
        } else if (frameLayout.getVisibility() == 4) {
            this.iplay_layout_arrow_down.setVisibility(0);
        }
        boolean canScrollVertically2 = this.iplay_recycler_mode.canScrollVertically(-1);
        FrameLayout frameLayout2 = this.iplay_layout_arrow_up;
        if (!canScrollVertically2) {
            frameLayout2.setVisibility(4);
        } else if (frameLayout2.getVisibility() == 4) {
            this.iplay_layout_arrow_up.setVisibility(0);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollVibrateHideImg() {
        boolean canScrollVertically = this.iplay_recycler_vibration.canScrollVertically(1);
        FrameLayout frameLayout = this.iplay_layout_vibrate_arrow_down;
        if (!canScrollVertically) {
            frameLayout.setVisibility(4);
        } else if (frameLayout.getVisibility() == 4) {
            this.iplay_layout_vibrate_arrow_down.setVisibility(0);
        }
        boolean canScrollVertically2 = this.iplay_recycler_vibration.canScrollVertically(-1);
        FrameLayout frameLayout2 = this.iplay_layout_vibrate_arrow_up;
        if (!canScrollVertically2) {
            frameLayout2.setVisibility(4);
        } else if (frameLayout2.getVisibility() == 4) {
            this.iplay_layout_vibrate_arrow_up.setVisibility(0);
        }
    }

    private void setListener() {
        this.iplay_img_close_left.setOnTouchListener(this.leftOnTouch);
        this.iplay_img_close_right.setOnTouchListener(this.rightOnTouch);
        this.iplay_layout_speed_safe.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.2
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlaySMVUIFragment.this.iplay_layout_speed_safe.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                PlaySMVUIFragment playSMVUIFragment = PlaySMVUIFragment.this;
                playSMVUIFragment.refreshSafeUI(playSMVUIFragment.mSafeMax);
            }
        });
        this.iplay_layout_vibrate_safe.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.3
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlaySMVUIFragment.this.iplay_layout_vibrate_safe.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                PlaySMVUIFragment playSMVUIFragment = PlaySMVUIFragment.this;
                playSMVUIFragment.refreshVibrateSafeUI(playSMVUIFragment.mVibrateSafeMax);
            }
        });
        this.iplay_layout_speed_safe.setOnTouchListener(this.speedSafeOnTouch);
        this.iplay_layout_vibrate_safe.setOnTouchListener(this.vibrateSafeOnTouch);
        this.iplay_img_mode.setOnClickListener(this);
        this.iplay_img_vibrate.setOnClickListener(this);
        this.iplay_img_music.setOnClickListener(this);
        this.iplay_layout_mode.setOnClickListener(this);
        this.iplay_layout_vibration.setOnClickListener(this);
        this.iplay_layout_mode_border.setOnClickListener(this);
        this.iplay_layout_vibration_border.setOnClickListener(this);
        this.iplay_layout_arrow_up.setOnClickListener(this);
        this.iplay_layout_arrow_down.setOnClickListener(this);
        this.iplay_layout_vibrate_arrow_up.setOnClickListener(this);
        this.iplay_layout_vibrate_arrow_down.setOnClickListener(this);
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.4
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMVUIFragment.this.scrollModeHideImg();
            }
        });
        this.iplay_recycler_vibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.5
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMVUIFragment.this.scrollVibrateHideImg();
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.6
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMVUIFragment.this.inAnim = false;
                if (i != PlaySMVUIFragment.this.isShowMode) {
                    PlaySMVUIFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMVUIFragment.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.7
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMVUIFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMVUIFragment.this.inAnim = true;
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
    public void stopAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 1.0f, 0.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.8
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMVUIFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != PlaySMVUIFragment.this.isShowMode) {
                    PlaySMVUIFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMVUIFragment.this.inAnim = true;
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
    public void initBleAAData(String str, String str2) {
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
                    Message message3 = new Message();
                    message3.what = 6;
                    message3.obj = 0;
                    this.mHandler.sendMessage(message3);
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            case "03":
            case "04":
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
                    if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() != 1 && ProductUtil.isSM2Type()) {
                        int intValue3 = Integer.valueOf(str2, 16).intValue();
                        Message message6 = new Message();
                        message6.what = 5;
                        message6.obj = Integer.valueOf(intValue3);
                        this.mHandler.sendMessage(message6);
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

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initBleCCData(String str, String str2, String str3) {
        int i;
        str2.hashCode();
        i = 0;
        switch (str2) {
            case "01":
                try {
                    Message message = new Message();
                    if (!"B0".equals(str3)) {
                        if ("B1".equals(str3)) {
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
                try {
                    int intValue = Integer.valueOf(str3, 16).intValue();
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
                    int intValue2 = Integer.valueOf(str3, 16).intValue();
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
                    int intValue3 = Integer.valueOf(str3, 16).intValue();
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
                    int intValue4 = Integer.valueOf(str3, 16).intValue();
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

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initProductData() {
        ProductModel productModel = MyApplication.getApplication().getmProductModel();
        if (productModel == null) {
            return;
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
        int music = productModel.getMusic();
        ImageView imageView = this.iplay_img_music;
        if (music == 1) {
            imageView.setVisibility(0);
        } else {
            imageView.setVisibility(8);
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment, com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        super.lazyLoad();
        init();
        setListener();
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
        if (id == R.id.iplay_img_music) {
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
        View inflate = layoutInflater.inflate(R.layout.fragment_iplay_smv_ui, viewGroup, false);
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
            if (socketSpeedModel == null || socketSpeedModel.getSuccess() != 1) {
                return;
            }
            if (!"setSpeed".equals(socketSpeedModel.getAction())) {
                if ("setMode".equals(socketSpeedModel.getAction())) {
                    if (socketSpeedModel.isMotor1Mode()) {
                        Message message = new Message();
                        message.what = 2;
                        message.obj = Integer.valueOf(socketSpeedModel.getMode());
                        this.mHandler.sendMessage(message);
                        return;
                    }
                    if (socketSpeedModel.isMotor2Mode()) {
                        Message message2 = new Message();
                        message2.what = 5;
                        message2.obj = Integer.valueOf(socketSpeedModel.getMode());
                        this.mHandler.sendMessage(message2);
                        return;
                    }
                    return;
                }
                return;
            }
            if (socketSpeedModel.getLines() != null && socketSpeedModel.getLines().isJsonObject()) {
                JsonObject jsonObject = (JsonObject) socketSpeedModel.getLines();
                HashMap hashMap = new HashMap();
                for (String str2 : jsonObject.keySet()) {
                    JsonObject asJsonObject = jsonObject.getAsJsonObject(str2);
                    hashMap.put(str2, new Point(asJsonObject.get("x").getAsInt(), asJsonObject.get("y").getAsInt()));
                    LogUtil.e("WebSocket line key:" + str2 + ", value:" + asJsonObject);
                }
                if (socketSpeedModel.isMotor2()) {
                    this.iplay_playView_vibrate.setPointArr(hashMap);
                } else {
                    this.iplay_playView.setPointArr(hashMap);
                }
            }
            final int speed = socketSpeedModel.getSpeed();
            if (speed > 100) {
                speed = 100;
            } else if (speed < 0) {
                speed = 0;
            }
            boolean isMotor2 = socketSpeedModel.isMotor2();
            BaseActivity baseActivity = this.baseActivity;
            if (isMotor2) {
                baseActivity.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.14
                    @Override // java.lang.Runnable
                    public void run() {
                        PlaySMVUIFragment.this.iplay_progress_vibrate.setSecondaryProgress(speed);
                    }
                });
                int i = this.mVibrateSafeMax;
                if (speed > i) {
                    speed = i;
                }
                Message message3 = new Message();
                message3.what = 7;
                message3.obj = Integer.valueOf(speed);
                this.mHandler.sendMessage(message3);
                return;
            }
            baseActivity.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.fragment.PlaySMVUIFragment.15
                @Override // java.lang.Runnable
                public void run() {
                    PlaySMVUIFragment.this.iplay_progress_speed.setSecondaryProgress(speed);
                }
            });
            int i2 = this.mSafeMax;
            if (speed > i2) {
                speed = i2;
            }
            Message message4 = new Message();
            message4.what = 1;
            message4.obj = Integer.valueOf(speed);
            this.mHandler.sendMessage(message4);
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
