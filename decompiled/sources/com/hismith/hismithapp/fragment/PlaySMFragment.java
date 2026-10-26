package com.hismith.hismithapp.fragment;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
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
import androidx.core.content.ContextCompat;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.SocketIplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.utils.BleSpeedUtil;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMFragment extends SocketPlayFragment implements View.OnClickListener, SensorEventListener {
    private BaseActivity baseActivity;
    private SocketIplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private BothWayArrowView mArrowBoth;
    private BleSpeedUtil mBleSpeedUtil;
    private ImageView mImgCloseMode;
    private ImageView mImgCloseVibration;
    private ImageView mImgGesture;
    private ImageView mImgHeart;
    private ImageView mImgMic;
    private ImageView mImgMode;
    private ImageView mImgShake;
    private ImageView mImgVibrate;
    private LinearLayout mLayoutArrow;
    private FrameLayout mLayoutArrowDown;
    private FrameLayout mLayoutArrowUp;
    private GridLayoutManager mLayoutManager;
    private FrameLayout mLayoutMicBtn;
    private FrameLayout mLayoutMode;
    private LinearLayout mLayoutModeBorder;
    private FrameLayout mLayoutModeBtn;
    private FrameLayout mLayoutShakeBtn;
    private FrameLayout mLayoutVibrateArrowDown;
    private FrameLayout mLayoutVibrateArrowUp;
    private FrameLayout mLayoutVibrateBtn;
    private FrameLayout mLayoutVibration;
    private LinearLayout mLayoutVibrationBorder;
    private List<String> mModes;
    private PlayView_Four mPlayView;
    private SocketProductModel mProductModel;
    private ProgressBar mProgressSpeed;
    private RecyclerView mRecyclerMode;
    private RecyclerView mRecyclerVibration;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private TextView mTxtMaxSpeed;
    private TextView mTxtMinSpeed;
    private TextView mTxtSliding;
    private TextView mTxtSpeed;
    private SocketIplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private List<String> mVibrationModes;
    private View mViewTouch;
    private RecorderVolumeManager recorderVolumeManager;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_VIBRATION_MODE = 5;
    private int leftAction = 1;
    private int rightAction = 1;
    private boolean isMicMode = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                int intValue = ((Integer) message.obj).intValue();
                PlaySMFragment.this.mTxtSpeed.setText(intValue + "");
                PlaySMFragment.this.mProgressSpeed.setProgress(intValue);
                PlaySMFragment.this.isShowMode = 0;
                if (PlaySMFragment.this.mLayoutMode.getVisibility() == 0) {
                    PlaySMFragment playSMFragment = PlaySMFragment.this;
                    playSMFragment.stopAnimation(playSMFragment.mLayoutMode);
                }
                PlaySMFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                if (PlaySMFragment.this.mAdapter != null) {
                    PlaySMFragment.this.mAdapter.setClickPosition(-1);
                }
            } else if (i == 2) {
                int intValue2 = ((Integer) message.obj).intValue();
                PlaySMFragment playSMFragment2 = PlaySMFragment.this;
                if (intValue2 <= 0) {
                    playSMFragment2.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                } else {
                    playSMFragment2.isShowMode = 1;
                    if (PlaySMFragment.this.mLayoutMode.getVisibility() != 0) {
                        PlaySMFragment playSMFragment3 = PlaySMFragment.this;
                        playSMFragment3.startAnimation(playSMFragment3.mLayoutMode, PlaySMFragment.this.mLayoutVibration);
                    }
                    PlaySMFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                }
                if (PlaySMFragment.this.mAdapter != null) {
                    PlaySMFragment.this.mAdapter.setClickPosition(intValue2 - 1);
                    if (intValue2 != 0) {
                        PlaySMFragment.this.mRecyclerMode.smoothScrollToPosition(intValue2);
                    }
                }
            } else if (i == 5) {
                int intValue3 = ((Integer) message.obj).intValue();
                PlaySMFragment playSMFragment4 = PlaySMFragment.this;
                if (intValue3 >= 1) {
                    playSMFragment4.isShowMode = 2;
                    if (PlaySMFragment.this.mLayoutVibration.getVisibility() != 0) {
                        PlaySMFragment playSMFragment5 = PlaySMFragment.this;
                        playSMFragment5.startAnimation(playSMFragment5.mLayoutVibration, PlaySMFragment.this.mLayoutMode);
                    }
                    PlaySMFragment.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                } else {
                    playSMFragment4.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                }
                if (PlaySMFragment.this.mVibrationAdapter != null) {
                    PlaySMFragment.this.mVibrationAdapter.setClickPosition(intValue3 - 1);
                    if (intValue3 != 0) {
                        PlaySMFragment.this.mRecyclerVibration.smoothScrollToPosition(intValue3);
                    }
                }
            }
            return false;
        }
    });
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private long triggerTime = 0;
    private int shakeCount = 0;

    private void closeSocket() {
        if (SocketManager.getInstance().getWebSocket() != null) {
            LogUtil.e("WebSocket 发送：关闭");
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public SocketProductModel getBinderProduct() {
        BaseActivity baseActivity = this.baseActivity;
        return baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getBinder().getSocketProductModel() : baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getBinder().getSocketProductModel() : new SocketProductModel();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideGestureTip() {
        if (this.mLayoutArrow.getVisibility() == 0) {
            this.mLayoutArrow.setVisibility(8);
            this.mArrowBoth.setOnValueListener(null);
        }
    }

    private void init() {
        Bundle arguments = getArguments();
        if (arguments != null) {
            this.mProductModel = (SocketProductModel) arguments.getSerializable("ProductModel");
        }
        this.recorderVolumeManager = new RecorderVolumeManager();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.mRecyclerMode.setLayoutManager(gridLayoutManager);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this.baseActivity, 2);
        this.mVibrationLayoutManager = gridLayoutManager2;
        this.mRecyclerVibration.setLayoutManager(gridLayoutManager2);
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        SocketIplayModeAdapter socketIplayModeAdapter = new SocketIplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = socketIplayModeAdapter;
        socketIplayModeAdapter.setOnItemClickListener(new SocketIplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.2
            @Override // com.hismith.hismithapp.adapter.SocketIplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                PlaySMFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                int i2 = i + 1;
                SocketManager.getInstance().setMode(i2);
                PlaySMFragment.this.getBinderProduct().setMode(i2);
            }
        });
        this.mRecyclerMode.setAdapter(this.mAdapter);
        SocketIplayModeAdapter socketIplayModeAdapter2 = new SocketIplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = socketIplayModeAdapter2;
        socketIplayModeAdapter2.setOnItemClickListener(new SocketIplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.3
            @Override // com.hismith.hismithapp.adapter.SocketIplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                PlaySMFragment.this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                int i2 = i + 1;
                SocketManager.getInstance().setVibrateMode(i2);
                PlaySMFragment.this.getBinderProduct().setVibration(i2);
            }
        });
        this.mRecyclerVibration.setAdapter(this.mVibrationAdapter);
        if (this.mProductModel != null) {
            PlayView_Four playView_Four = this.mPlayView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(r0.getSlide());
            }
            initProductData();
            this.mTxtSpeed.setText(this.mProductModel.getSpeed() + "");
            this.mProgressSpeed.setProgress(this.mProductModel.getSpeed());
            int mode = this.mProductModel.getMode();
            ImageView imageView = this.mImgMode;
            if (mode > 0) {
                imageView.setImageResource(R.mipmap.ic_iplay_mode_select);
                SocketIplayModeAdapter socketIplayModeAdapter3 = this.mAdapter;
                if (socketIplayModeAdapter3 != null) {
                    socketIplayModeAdapter3.setClickPosition(this.mProductModel.getMode() - 1);
                }
            } else {
                imageView.setImageResource(R.mipmap.ic_iplay_mode);
            }
            int vibration = this.mProductModel.getVibration();
            ImageView imageView2 = this.mImgVibrate;
            if (vibration > 0) {
                imageView2.setImageResource(R.mipmap.ic_iplay_vibration_select);
                SocketIplayModeAdapter socketIplayModeAdapter4 = this.mVibrationAdapter;
                if (socketIplayModeAdapter4 != null) {
                    socketIplayModeAdapter4.setClickPosition(this.mProductModel.getVibration() - 1);
                }
            } else {
                imageView2.setImageResource(R.mipmap.ic_iplay_vibration);
            }
        }
        scrollModeHideImg();
        scrollVibrateHideImg();
        initSensor();
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

    private void initRecordPermission() {
        if (ContextCompat.checkSelfPermission(this.baseActivity, "android.permission.RECORD_AUDIO") == 0) {
            startMicClick();
            return;
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                PlaySMFragment.this.m599x1d7b434f((Permission) obj);
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
        this.mSensor = sensorManager.getDefaultSensor(1);
        this.mBleSpeedUtil = new BleSpeedUtil(this.baseActivity);
    }

    private void initView(View view) {
        this.mPlayView = (PlayView_Four) view.findViewById(R.id.iplay_playView);
        this.mProgressSpeed = (ProgressBar) view.findViewById(R.id.iplay_progress_speed);
        this.mTxtMaxSpeed = (TextView) view.findViewById(R.id.iplay_txt_max_speed);
        this.mTxtMinSpeed = (TextView) view.findViewById(R.id.iplay_txt_min_speed);
        this.mTxtSpeed = (TextView) view.findViewById(R.id.iplay_txt_speed);
        this.mLayoutModeBtn = (FrameLayout) view.findViewById(R.id.iplay_layout_mode_btn);
        this.mLayoutVibrateBtn = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_btn);
        this.mLayoutMicBtn = (FrameLayout) view.findViewById(R.id.iplay_layout_mic_btn);
        this.mLayoutShakeBtn = (FrameLayout) view.findViewById(R.id.iplay_layout_shake_btn);
        this.mImgMode = (ImageView) view.findViewById(R.id.iplay_img_mode);
        this.mImgVibrate = (ImageView) view.findViewById(R.id.iplay_img_vibrate);
        this.mImgMic = (ImageView) view.findViewById(R.id.iplay_img_mic);
        this.mImgShake = (ImageView) view.findViewById(R.id.iplay_img_shake);
        this.mLayoutMode = (FrameLayout) view.findViewById(R.id.iplay_layout_mode);
        this.mLayoutModeBorder = (LinearLayout) view.findViewById(R.id.iplay_layout_mode_border);
        this.mRecyclerMode = (RecyclerView) view.findViewById(R.id.iplay_recycler_mode);
        this.mLayoutVibration = (FrameLayout) view.findViewById(R.id.iplay_layout_vibration);
        this.mLayoutVibrationBorder = (LinearLayout) view.findViewById(R.id.iplay_layout_vibration_border);
        this.mRecyclerVibration = (RecyclerView) view.findViewById(R.id.iplay_recycler_vibration);
        this.mImgHeart = (ImageView) view.findViewById(R.id.iplay_img_heart);
        this.mImgCloseMode = (ImageView) view.findViewById(R.id.iplay_img_close_mode);
        this.mImgCloseVibration = (ImageView) view.findViewById(R.id.iplay_img_close_vibration);
        this.mLayoutArrowUp = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_up);
        this.mLayoutArrowDown = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_down);
        this.mLayoutVibrateArrowUp = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.mLayoutVibrateArrowDown = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.mViewTouch = view.findViewById(R.id.view_touch);
        this.mLayoutArrow = (LinearLayout) view.findViewById(R.id.layout_arrow);
        this.mArrowBoth = (BothWayArrowView) view.findViewById(R.id.arrow_both);
        this.mImgGesture = (ImageView) view.findViewById(R.id.img_gesture);
        TextView textView = (TextView) view.findViewById(R.id.txt_sliding);
        this.mTxtSliding = textView;
        LanguageUtil.setText(this.baseActivity, textView, "slidingControlSpeed");
        this.mPlayView.setProgressBar(this.mProgressSpeed);
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
        this.mLayoutModeBtn.setOnClickListener(this);
        this.mLayoutVibrateBtn.setOnClickListener(this);
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
        this.mViewTouch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                PlaySMFragment.this.hideGestureTip();
                return false;
            }
        });
        this.mRecyclerMode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.5
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMFragment.this.scrollModeHideImg();
            }
        });
        this.mRecyclerVibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.6
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMFragment.this.scrollVibrateHideImg();
            }
        });
        this.mPlayView.setOnLineChangeListener(new PlayView_Four.OnLineChangeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.7
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
            public void onLineChange(int i, JsonObject jsonObject) {
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != PlaySMFragment.this.mProgressSpeed.getProgress() || i == 0) {
                    PlaySMFragment.this.mTxtSpeed.setText(i + "");
                    PlaySMFragment.this.mProgressSpeed.setProgress(i);
                    if (PlaySMFragment.this.mAdapter.getClickPosition() != -1) {
                        PlaySMFragment.this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
                        PlaySMFragment.this.mAdapter.setClickPosition(-1);
                    }
                }
                SocketManager.getInstance().setSpeed(i, jsonObject);
                PlaySMFragment.this.getBinderProduct().setSpeed(i);
                PlaySMFragment.this.getBinderProduct().setMode(0);
            }
        });
        this.mArrowBoth.setOnValueListener(new BothWayArrowView.OnValueListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.8
            @Override // com.hismith.hismithapp.view.BothWayArrowView.OnValueListener
            public void onValueChange(float f) {
                PlaySMFragment.this.mImgGesture.setRotation(f * (-45.0f));
            }
        });
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.9
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
                Message message = new Message();
                message.what = 1;
                message.obj = Integer.valueOf(i);
                PlaySMFragment.this.mHandler.sendMessage(message);
                SocketManager.getInstance().setSpeed(i);
                PlaySMFragment.this.getBinderProduct().setSpeed(i);
                PlaySMFragment.this.getBinderProduct().setMode(0);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.10
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMFragment.this.inAnim = false;
                if (i != PlaySMFragment.this.isShowMode) {
                    PlaySMFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMFragment.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.11
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet2 = new AnimatorSet();
        this.mAnimatorSet = animatorSet2;
        animatorSet2.play(ofFloat3).with(ofFloat4);
        this.mAnimatorSet.play(ofFloat3).before(ofFloat);
        this.mAnimatorSet.play(ofFloat).with(ofFloat2);
        this.mAnimatorSet.start();
    }

    private void startHeart() {
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
            return;
        }
        imageView.setImageResource(R.mipmap.ic_iplay_mic);
        this.recorderVolumeManager.stopRecording();
        SocketManager.getInstance().setSpeed(0);
        getBinderProduct().setSpeed(0);
        getBinderProduct().setMode(0);
        Message message = new Message();
        message.what = 1;
        message.obj = 0;
        this.mHandler.sendMessage(message);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.12
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != PlaySMFragment.this.isShowMode) {
                    PlaySMFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    private void stopHeart() {
        if (this.mImgHeart.getVisibility() == 0) {
            this.mImgHeart.clearAnimation();
            this.mImgHeart.setVisibility(8);
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initProductData() {
        SocketProductModel socketProductModel = this.mProductModel;
        if (socketProductModel == null || socketProductModel.getModesUrl() == null) {
            return;
        }
        this.mModes.clear();
        if (this.mProductModel.getModesUrl().getModes() != null) {
            this.mModes.addAll(Arrays.asList(this.mProductModel.getModesUrl().getModes()));
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
        this.mVibrationModes.clear();
        String[] vibrations = this.mProductModel.getModesUrl().getVibrations();
        FrameLayout frameLayout = this.mLayoutVibrateBtn;
        if (vibrations != null) {
            frameLayout.setVisibility(0);
            this.mVibrationModes.addAll(Arrays.asList(this.mProductModel.getModesUrl().getVibrations()));
        } else {
            frameLayout.setVisibility(8);
        }
        this.mVibrationAdapter.notifyDataSetChanged();
        scrollVibrateHideImg();
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-fragment-PlaySMFragment, reason: not valid java name */
    /* synthetic */ void m599x1d7b434f(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            startMicClick();
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.13
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + PlaySMFragment.this.baseActivity.getPackageName()));
                    PlaySMFragment.this.startActivity(intent);
                    DialogUtil.dismissDialog();
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment, com.hismith.hismithapp.fragment.LazyLoadFragment
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
            int progress = this.mProgressSpeed.getProgress();
            SocketManager.getInstance().setSpeed(progress);
            getBinderProduct().setSpeed(progress);
            getBinderProduct().setMode(0);
            this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
            SocketIplayModeAdapter socketIplayModeAdapter = this.mAdapter;
            if (socketIplayModeAdapter != null) {
                socketIplayModeAdapter.setClickPosition(-1);
                return;
            }
            return;
        }
        if (id == R.id.iplay_img_close_vibration) {
            SocketManager.getInstance().setVibrateMode(0);
            getBinderProduct().setVibration(0);
            this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
            SocketIplayModeAdapter socketIplayModeAdapter2 = this.mVibrationAdapter;
            if (socketIplayModeAdapter2 != null) {
                socketIplayModeAdapter2.setClickPosition(-1);
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
        View inflate = layoutInflater.inflate(R.layout.fragment_iplay, viewGroup, false);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.cancelRecording();
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onNotify(String str) {
        LogUtil.e("PlaySMFragment onNotify:" + str);
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
            SocketProductModel socketProductModel = this.mProductModel;
            this.mBleSpeedUtil.shakeSpeed(f, f2, f3, (socketProductModel == null || socketProductModel.getShake() <= 0) ? 100 : this.mProductModel.getShake(), new BleSpeedUtil.OnShakeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMFragment.14
                @Override // com.hismith.hismithapp.utils.BleSpeedUtil.OnShakeListener
                public void onShake(int i) {
                    Message message = new Message();
                    message.what = 1;
                    message.obj = Integer.valueOf(i);
                    PlaySMFragment.this.mHandler.sendMessage(message);
                    SocketManager.getInstance().setSpeed(i);
                    PlaySMFragment.this.getBinderProduct().setSpeed(i);
                    PlaySMFragment.this.getBinderProduct().setMode(0);
                }
            });
            if ((Math.abs(f) >= 20.0f || Math.abs(f2) >= 20.0f) && MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                if (this.triggerTime == currentTimeMillis) {
                    int i = this.shakeCount + 1;
                    this.shakeCount = i;
                    if (i >= 3) {
                        LogUtil.e("Sensor value 摇晃");
                        SocketManager.getInstance().setClimax();
                    }
                } else {
                    this.shakeCount = 0;
                }
                this.triggerTime = currentTimeMillis;
            }
        }
    }
}
