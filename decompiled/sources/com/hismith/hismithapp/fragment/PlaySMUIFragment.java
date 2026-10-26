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
import android.view.animation.ScaleAnimation;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.media3.exoplayer.Renderer;
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
import com.hismith.hismithapp.http.SocketManager;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMUIFragment extends SocketPlayFragment implements View.OnClickListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private BaseActivity baseActivity;
    private float downY;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private BleDeviceModel mBleDeviceModel;
    private Gson mGson;
    private ImageView mImgCloseLeft;
    private ImageView mImgCloseRight;
    private ImageView mImgHeart;
    private ImageView mImgMode;
    private ImageView mImgMusic;
    private ImageView mImgVibrate;
    private FrameLayout mLayoutArrowDown;
    private FrameLayout mLayoutArrowUp;
    private GridLayoutManager mLayoutManager;
    private FrameLayout mLayoutMode;
    private LinearLayout mLayoutModeBorder;
    private FrameLayout mLayoutModeBtn;
    private FrameLayout mLayoutMusicBtn;
    private ViewGroup mLayoutSafeSpeed;
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
    private int mSafeMax;
    private TextView mTxtMaxSpeed;
    private TextView mTxtMinSpeed;
    private TextView mTxtSafeSpeed;
    private TextView mTxtSpeed;
    private IplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private List<ProductModel.Mode> mVibrationModes;
    private View mViewTouch;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_CLIMAX = 3;
    private final int WHAT_HIDE_HEART = 4;
    private final int WHAT_VIBRATION_MODE = 5;
    private final int WHAT_MUSIC = 6;
    private final int WHAT_SAFE = 7;
    private int leftAction = 1;
    private int rightAction = 1;
    private int music = -1;
    private boolean isMicMode = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        /* JADX WARN: Code restructure failed: missing block: B:45:0x018e, code lost:
        
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
                    PlaySMUIFragment.this.mTxtSpeed.setText(intValue + "");
                    PlaySMUIFragment.this.mProgressSpeed.setProgress(intValue);
                    PlaySMUIFragment.this.isShowMode = 0;
                    if (PlaySMUIFragment.this.mLayoutMode.getVisibility() == 0) {
                        PlaySMUIFragment playSMUIFragment = PlaySMUIFragment.this;
                        playSMUIFragment.stopAnimation(playSMUIFragment.mLayoutMode);
                    }
                    PlaySMUIFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (PlaySMUIFragment.this.mAdapter != null) {
                        PlaySMUIFragment.this.mAdapter.setClickPosition(-1);
                        break;
                    }
                    break;
                case 2:
                    int intValue2 = ((Integer) message.obj).intValue();
                    PlaySMUIFragment playSMUIFragment2 = PlaySMUIFragment.this;
                    if (intValue2 <= 0) {
                        playSMUIFragment2.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                    } else {
                        playSMUIFragment2.isShowMode = 1;
                        if (PlaySMUIFragment.this.mLayoutMode.getVisibility() != 0) {
                            PlaySMUIFragment playSMUIFragment3 = PlaySMUIFragment.this;
                            playSMUIFragment3.startAnimation(playSMUIFragment3.mLayoutMode, PlaySMUIFragment.this.mLayoutVibration);
                        }
                        PlaySMUIFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                    }
                    if (PlaySMUIFragment.this.mAdapter != null) {
                        PlaySMUIFragment.this.mAdapter.setClickPosition(intValue2 - 1);
                        if (intValue2 != 0) {
                            PlaySMUIFragment.this.mRecyclerMode.smoothScrollToPosition(intValue2);
                            break;
                        }
                    }
                    break;
                case 3:
                    if (message.obj != null) {
                        int intValue3 = ((Integer) message.obj).intValue();
                        PlaySMUIFragment playSMUIFragment4 = PlaySMUIFragment.this;
                        if (intValue3 != 1) {
                            playSMUIFragment4.stopHeart();
                            break;
                        } else {
                            playSMUIFragment4.startHeart();
                            PlaySMUIFragment.this.mHandler.sendEmptyMessageDelayed(4, Renderer.DEFAULT_DURATION_TO_PROGRESS_US);
                            break;
                        }
                    }
                    break;
                case 4:
                    PlaySMUIFragment.this.stopHeart();
                    break;
                case 5:
                    int intValue4 = ((Integer) message.obj).intValue();
                    PlaySMUIFragment playSMUIFragment5 = PlaySMUIFragment.this;
                    if (intValue4 >= 1) {
                        playSMUIFragment5.isShowMode = 2;
                        if (PlaySMUIFragment.this.mLayoutVibration.getVisibility() != 0) {
                            PlaySMUIFragment playSMUIFragment6 = PlaySMUIFragment.this;
                            playSMUIFragment6.startAnimation(playSMUIFragment6.mLayoutVibration, PlaySMUIFragment.this.mLayoutMode);
                        }
                        PlaySMUIFragment.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                    } else {
                        playSMUIFragment5.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                    }
                    if (PlaySMUIFragment.this.mVibrationAdapter != null) {
                        PlaySMUIFragment.this.mVibrationAdapter.setClickPosition(intValue4 - 1);
                        if (intValue4 != 0) {
                            PlaySMUIFragment.this.mRecyclerVibration.smoothScrollToPosition(intValue4);
                            break;
                        }
                    }
                    break;
                case 6:
                    int intValue5 = ((Integer) message.obj).intValue();
                    PlaySMUIFragment.this.music = intValue5;
                    PlaySMUIFragment playSMUIFragment7 = PlaySMUIFragment.this;
                    if (intValue5 <= 0) {
                        playSMUIFragment7.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice);
                        break;
                    } else {
                        playSMUIFragment7.mImgMusic.setImageResource(R.mipmap.ic_iplay_voice_select);
                        break;
                    }
                case 7:
                    PlaySMUIFragment.this.refreshSafeUI(((Integer) message.obj).intValue());
                    break;
            }
        }
    });
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.5
        /* JADX WARN: Code restructure failed: missing block: B:7:0x000d, code lost:
        
            if (r0 != 3) goto L28;
         */
        @Override // android.view.View.OnTouchListener
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action != 0) {
                if (action != 1) {
                    if (action == 2) {
                        int top = (int) (view.getTop() + (motionEvent.getY() - PlaySMUIFragment.this.downY));
                        if (top < 0) {
                            top = 0;
                        } else if (top > PlaySMUIFragment.this.mProgressSpeed.getHeight()) {
                            top = PlaySMUIFragment.this.mProgressSpeed.getHeight();
                        }
                        int height = view.getHeight() + top;
                        LogUtil.e("onTouch  t:" + top + ", b:" + height + ", offset:" + (height - top));
                        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) PlaySMUIFragment.this.mLayoutSafeSpeed.getLayoutParams();
                        layoutParams.setMargins(0, top, 0, 0);
                        PlaySMUIFragment.this.mLayoutSafeSpeed.setLayoutParams(layoutParams);
                        int height2 = 100 - (((int) (top * 100.0d)) / PlaySMUIFragment.this.mProgressSpeed.getHeight());
                        PlaySMUIFragment.this.mTxtSafeSpeed.setText(height2 + "");
                        PlaySMUIFragment.this.mSafeMax = height2;
                        PlaySMUIFragment playSMUIFragment = PlaySMUIFragment.this;
                        playSMUIFragment.onSetSafeMax(playSMUIFragment.mSafeMax);
                        if (height2 >= 0 && height2 < PlaySMUIFragment.this.mProgressSpeed.getSecondaryProgress()) {
                            BleManager.getInstance(PlaySMUIFragment.this.baseActivity).writeSpeed(height2);
                            PlaySMUIFragment.this.mProgressSpeed.setProgress(height2);
                            PlaySMUIFragment.this.mTxtSpeed.setText(height2 + "");
                        } else if (height2 >= PlaySMUIFragment.this.mProgressSpeed.getSecondaryProgress() && PlaySMUIFragment.this.mProgressSpeed.getSecondaryProgress() != PlaySMUIFragment.this.mProgressSpeed.getProgress()) {
                            BleManager.getInstance(PlaySMUIFragment.this.baseActivity).writeSpeed(PlaySMUIFragment.this.mProgressSpeed.getSecondaryProgress());
                            PlaySMUIFragment.this.mProgressSpeed.setProgress(PlaySMUIFragment.this.mProgressSpeed.getSecondaryProgress());
                            PlaySMUIFragment.this.mTxtSpeed.setText(PlaySMUIFragment.this.mProgressSpeed.getSecondaryProgress() + "");
                        }
                    }
                }
                LogUtil.e("onTouch 松开Y:" + motionEvent.getY());
            } else {
                PlaySMUIFragment.this.downY = motionEvent.getY();
                LogUtil.e("onTouch 按下Y:" + motionEvent.getY());
                LogUtil.e("onTouch top:" + view.getTop());
            }
            return true;
        }
    };
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.6
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMUIFragment.this.leftAction = 0;
                PlaySMUIFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMUIFragment.this.lastLeftClickTime;
                PlaySMUIFragment playSMUIFragment = PlaySMUIFragment.this;
                if (j < 500) {
                    boolean z = playSMUIFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMUIFragment playSMUIFragment2 = PlaySMUIFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMUIFragment2.baseActivity).exit();
                    } else if (playSMUIFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMUIFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMUIFragment.lastLeftClickTime = currentTimeMillis;
                PlaySMUIFragment.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.7
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMUIFragment.this.rightAction = 0;
                PlaySMUIFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMUIFragment.this.lastRightClickTime;
                PlaySMUIFragment playSMUIFragment = PlaySMUIFragment.this;
                if (j < 500) {
                    boolean z = playSMUIFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMUIFragment playSMUIFragment2 = PlaySMUIFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMUIFragment2.baseActivity).exit();
                    } else if (playSMUIFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMUIFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMUIFragment.lastRightClickTime = currentTimeMillis;
                PlaySMUIFragment.this.rightAction = 1;
            }
            return true;
        }
    };
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.12
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                PlaySMUIFragment.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("AA".equals(stringExtra)) {
                    PlaySMUIFragment.this.initBleAAData(stringExtra2, stringExtra3);
                } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    PlaySMUIFragment.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };

    public PlaySMUIFragment(int i) {
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
            shakeAnimation(this.mImgCloseLeft);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            LogUtil.e("onTouch 右抖动");
            shakeAnimation(this.mImgCloseRight);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
    }

    private void init() {
        this.mGson = new Gson();
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
        this.mRecyclerMode.setAdapter(iplayModeAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        this.mRecyclerVibration.setAdapter(iplayModeAdapter2);
        initProductData();
        BleDeviceModel bleDeviceModel = BleManager.getInstance(this.baseActivity).getmConnBleDevice();
        this.mBleDeviceModel = bleDeviceModel;
        if (bleDeviceModel != null) {
            this.mTxtSpeed.setText(this.mBleDeviceModel.getSpeed() + "");
            this.mProgressSpeed.setProgress(this.mBleDeviceModel.getSpeed());
            int mode = this.mBleDeviceModel.getMode();
            ImageView imageView = this.mImgMode;
            if (mode > 0) {
                imageView.setImageResource(R.mipmap.ic_iplay_mode_select);
                IplayModeAdapter iplayModeAdapter3 = this.mAdapter;
                if (iplayModeAdapter3 != null) {
                    iplayModeAdapter3.setClickPosition(this.mBleDeviceModel.getMode() - 1);
                }
            } else {
                imageView.setImageResource(R.mipmap.ic_iplay_mode);
            }
            int vibrationMode = this.mBleDeviceModel.getVibrationMode();
            ImageView imageView2 = this.mImgVibrate;
            if (vibrationMode > 0) {
                imageView2.setImageResource(R.mipmap.ic_iplay_vibration_select);
                IplayModeAdapter iplayModeAdapter4 = this.mVibrationAdapter;
                if (iplayModeAdapter4 != null) {
                    iplayModeAdapter4.setClickPosition(this.mBleDeviceModel.getVibrationMode() - 1);
                }
            } else {
                imageView2.setImageResource(R.mipmap.ic_iplay_vibration);
            }
            int music = this.mBleDeviceModel.getMusic();
            this.music = music;
            ImageView imageView3 = this.mImgMusic;
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

    private void initView(View view) {
        this.mPlayView = (PlayView_Four) view.findViewById(R.id.iplay_playView);
        this.mProgressSpeed = (ProgressBar) view.findViewById(R.id.iplay_progress_speed);
        this.mTxtMaxSpeed = (TextView) view.findViewById(R.id.iplay_txt_max_speed);
        this.mTxtMinSpeed = (TextView) view.findViewById(R.id.iplay_txt_min_speed);
        this.mTxtSpeed = (TextView) view.findViewById(R.id.iplay_txt_speed);
        this.mLayoutModeBtn = (FrameLayout) view.findViewById(R.id.iplay_layout_mode_btn);
        this.mLayoutVibrateBtn = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_btn);
        this.mLayoutMusicBtn = (FrameLayout) view.findViewById(R.id.iplay_layout_music_btn);
        this.mImgMode = (ImageView) view.findViewById(R.id.iplay_img_mode);
        this.mImgVibrate = (ImageView) view.findViewById(R.id.iplay_img_vibrate);
        this.mImgMusic = (ImageView) view.findViewById(R.id.iplay_img_music);
        this.mImgCloseLeft = (ImageView) view.findViewById(R.id.iplay_img_close_left);
        this.mImgCloseRight = (ImageView) view.findViewById(R.id.iplay_img_close_right);
        this.mLayoutMode = (FrameLayout) view.findViewById(R.id.iplay_layout_mode);
        this.mLayoutModeBorder = (LinearLayout) view.findViewById(R.id.iplay_layout_mode_border);
        this.mRecyclerMode = (RecyclerView) view.findViewById(R.id.iplay_recycler_mode);
        this.mLayoutVibration = (FrameLayout) view.findViewById(R.id.iplay_layout_vibration);
        this.mLayoutVibrationBorder = (LinearLayout) view.findViewById(R.id.iplay_layout_vibration_border);
        this.mRecyclerVibration = (RecyclerView) view.findViewById(R.id.iplay_recycler_vibration);
        this.mImgHeart = (ImageView) view.findViewById(R.id.iplay_img_heart);
        this.mLayoutArrowUp = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_up);
        this.mLayoutArrowDown = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_down);
        this.mLayoutVibrateArrowUp = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.mLayoutVibrateArrowDown = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.mLayoutSafeSpeed = (ViewGroup) view.findViewById(R.id.iplay_layout_safe_speed);
        this.mTxtSafeSpeed = (TextView) view.findViewById(R.id.iplay_txt_safe_speed);
        this.mViewTouch = view.findViewById(R.id.view_touch);
        this.mPlayView.setProgressBar(this.mProgressSpeed);
        this.mPlayView.setTouch(false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.mTxtSafeSpeed.setText(i + "");
        double height = this.mProgressSpeed.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.mLayoutSafeSpeed.getLayoutParams();
        layoutParams.setMargins(0, i2, 0, 0);
        this.mLayoutSafeSpeed.setLayoutParams(layoutParams);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        boolean canScrollVertically = this.mRecyclerMode.canScrollVertically(1);
        FrameLayout frameLayout = this.mLayoutArrowDown;
        if (!canScrollVertically) {
            frameLayout.setVisibility(4);
        } else if (frameLayout.getVisibility() == 4) {
            this.mLayoutArrowDown.setVisibility(0);
        }
        boolean canScrollVertically2 = this.mRecyclerMode.canScrollVertically(-1);
        FrameLayout frameLayout2 = this.mLayoutArrowUp;
        if (!canScrollVertically2) {
            frameLayout2.setVisibility(4);
        } else if (frameLayout2.getVisibility() == 4) {
            this.mLayoutArrowUp.setVisibility(0);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollVibrateHideImg() {
        boolean canScrollVertically = this.mRecyclerVibration.canScrollVertically(1);
        FrameLayout frameLayout = this.mLayoutVibrateArrowDown;
        if (!canScrollVertically) {
            frameLayout.setVisibility(4);
        } else if (frameLayout.getVisibility() == 4) {
            this.mLayoutVibrateArrowDown.setVisibility(0);
        }
        boolean canScrollVertically2 = this.mRecyclerVibration.canScrollVertically(-1);
        FrameLayout frameLayout2 = this.mLayoutVibrateArrowUp;
        if (!canScrollVertically2) {
            frameLayout2.setVisibility(4);
        } else if (frameLayout2.getVisibility() == 4) {
            this.mLayoutVibrateArrowUp.setVisibility(0);
        }
    }

    private void setListener() {
        this.mImgCloseLeft.setOnTouchListener(this.leftOnTouch);
        this.mImgCloseRight.setOnTouchListener(this.rightOnTouch);
        this.mLayoutSafeSpeed.setOnTouchListener(this.safeOnTouch);
        this.mLayoutSafeSpeed.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.2
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlaySMUIFragment.this.mLayoutSafeSpeed.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                PlaySMUIFragment playSMUIFragment = PlaySMUIFragment.this;
                playSMUIFragment.refreshSafeUI(playSMUIFragment.mSafeMax);
            }
        });
        this.mLayoutModeBtn.setOnClickListener(this);
        this.mLayoutVibrateBtn.setOnClickListener(this);
        this.mLayoutMusicBtn.setOnClickListener(this);
        this.mLayoutMode.setOnClickListener(this);
        this.mLayoutVibration.setOnClickListener(this);
        this.mLayoutModeBorder.setOnClickListener(this);
        this.mLayoutVibrationBorder.setOnClickListener(this);
        this.mLayoutArrowUp.setOnClickListener(this);
        this.mLayoutArrowDown.setOnClickListener(this);
        this.mLayoutVibrateArrowUp.setOnClickListener(this);
        this.mLayoutVibrateArrowDown.setOnClickListener(this);
        this.mRecyclerMode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.3
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMUIFragment.this.scrollModeHideImg();
            }
        });
        this.mRecyclerVibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.4
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMUIFragment.this.scrollVibrateHideImg();
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.8
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMUIFragment.this.inAnim = false;
                if (i != PlaySMUIFragment.this.isShowMode) {
                    PlaySMUIFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMUIFragment.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.9
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMUIFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMUIFragment.this.inAnim = true;
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.10
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMUIFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != PlaySMUIFragment.this.isShowMode) {
                    PlaySMUIFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMUIFragment.this.inAnim = true;
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

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initBleAAData(String str, String str2) {
        int i;
        str.hashCode();
        i = 0;
        switch (str) {
            case "02":
                this.mHandler.sendEmptyMessage(4);
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
                this.mHandler.sendEmptyMessage(4);
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
                this.mHandler.sendEmptyMessage(4);
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
                    if (MyApplication.getApplication().getmProductModel() != null) {
                        if (MyApplication.getApplication().getmProductModel().getClimax() != 1) {
                            if (Constans.DEVICE_TYPE_SM2.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
                                int intValue3 = Integer.valueOf(str2, 16).intValue();
                                Message message6 = new Message();
                                message6.what = 5;
                                message6.obj = Integer.valueOf(intValue3);
                                this.mHandler.sendMessage(message6);
                                break;
                            }
                        } else if ("00".equals(str2)) {
                            this.mHandler.sendEmptyMessage(3);
                            break;
                        }
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
            case "06":
                try {
                    int intValue3 = Integer.valueOf(str3, 16).intValue();
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
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment, com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
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
        View inflate = layoutInflater.inflate(R.layout.fragment_iplay_ui, viewGroup, false);
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
                if (!"setMode".equals(socketSpeedModel.getAction())) {
                    if ("setClimax".equals(socketSpeedModel.getAction())) {
                        Message message = new Message();
                        message.what = 3;
                        message.obj = Integer.valueOf(socketSpeedModel.getClimax());
                        this.mHandler.sendMessage(message);
                        return;
                    }
                    return;
                }
                if (socketSpeedModel.isMotor1Mode()) {
                    Message message2 = new Message();
                    message2.what = 2;
                    message2.obj = Integer.valueOf(socketSpeedModel.getMode());
                    this.mHandler.sendMessage(message2);
                    return;
                }
                if (socketSpeedModel.isMotor2Mode()) {
                    Message message3 = new Message();
                    message3.what = 5;
                    message3.obj = Integer.valueOf(socketSpeedModel.getMode());
                    this.mHandler.sendMessage(message3);
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
                this.mPlayView.setPointArr(hashMap);
            }
            final int speed = socketSpeedModel.getSpeed();
            if (speed > 100) {
                speed = 100;
            } else if (speed < 0) {
                speed = 0;
            }
            this.baseActivity.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.fragment.PlaySMUIFragment.11
                @Override // java.lang.Runnable
                public void run() {
                    PlaySMUIFragment.this.mProgressSpeed.setSecondaryProgress(speed);
                }
            });
            int i = this.mSafeMax;
            if (speed > i) {
                speed = i;
            }
            Message message4 = new Message();
            message4.what = 1;
            message4.obj = Integer.valueOf(speed);
            this.mHandler.sendMessage(message4);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onPause() {
        super.onPause();
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
    }

    public void shakeAnimation(View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(this.baseActivity, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }
}
