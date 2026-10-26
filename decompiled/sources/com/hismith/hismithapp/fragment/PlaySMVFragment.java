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
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
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
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.BleSpeedUtil;
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
public class PlaySMVFragment extends SocketPlayFragment implements View.OnClickListener, SensorEventListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private BothWayArrowView arrow_both;
    private BaseActivity baseActivity;
    private ImageView img_gesture;
    private ImageView iplay_img_close_left;
    private ImageView iplay_img_close_mode;
    private ImageView iplay_img_close_right;
    private ImageView iplay_img_close_vibration;
    private ImageView iplay_img_gesture_left;
    private ImageView iplay_img_gesture_right;
    private ImageView iplay_img_mic;
    private ImageView iplay_img_mic_vibrate;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_shake;
    private ImageView iplay_img_shake_vibrate;
    private ImageView iplay_img_vibrate;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
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
    private TextView iplay_txt_gesture_tip;
    private TextView iplay_txt_max_speed;
    private TextView iplay_txt_max_vibrate;
    private TextView iplay_txt_min_speed;
    private TextView iplay_txt_min_vibrate;
    private TextView iplay_txt_speed;
    private TextView iplay_txt_vibrate;
    private LinearLayout layout_arrow;
    private SocketIplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private BleSpeedUtil mBleSpeedUtil;
    private BleSpeedUtil mBleVibrateUtil;
    private GridLayoutManager mLayoutManager;
    private List<String> mModes;
    private SocketProductModel mProductModel;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private SocketIplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private List<String> mVibrationModes;
    private RecorderVolumeManager recorderVolumeManager;
    private TextView txt_sliding;
    private View view_touch;
    private boolean isMicMode = false;
    private boolean isMicVibrateMode = false;
    private int leftAction = 1;
    private int rightAction = 1;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_VIBRATE = 3;
    private final int WHAT_VIBRATION_MODE = 4;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                int intValue = ((Integer) message.obj).intValue();
                PlaySMVFragment.this.iplay_txt_speed.setText(intValue + "");
                PlaySMVFragment.this.iplay_progress_speed.setProgress(intValue);
                PlaySMVFragment.this.isShowMode = 0;
                if (PlaySMVFragment.this.iplay_layout_mode.getVisibility() == 0) {
                    PlaySMVFragment playSMVFragment = PlaySMVFragment.this;
                    playSMVFragment.stopAnimation(playSMVFragment.iplay_layout_mode);
                }
                PlaySMVFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                if (PlaySMVFragment.this.mAdapter != null) {
                    PlaySMVFragment.this.mAdapter.setClickPosition(-1);
                }
            } else if (i == 2) {
                int intValue2 = ((Integer) message.obj).intValue();
                PlaySMVFragment playSMVFragment2 = PlaySMVFragment.this;
                if (intValue2 <= 0) {
                    playSMVFragment2.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                } else {
                    playSMVFragment2.isShowMode = 1;
                    if (PlaySMVFragment.this.iplay_layout_mode.getVisibility() != 0) {
                        PlaySMVFragment playSMVFragment3 = PlaySMVFragment.this;
                        playSMVFragment3.startAnimation(playSMVFragment3.iplay_layout_mode, PlaySMVFragment.this.iplay_layout_vibration);
                    }
                    PlaySMVFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                }
                if (PlaySMVFragment.this.mAdapter != null) {
                    int i2 = intValue2 - 1;
                    PlaySMVFragment.this.mAdapter.setClickPosition(i2);
                    if (intValue2 > 0) {
                        PlaySMVFragment.this.iplay_recycler_mode.smoothScrollToPosition(i2);
                    }
                }
            } else if (i == 3) {
                int intValue3 = ((Integer) message.obj).intValue();
                PlaySMVFragment.this.iplay_txt_vibrate.setText(intValue3 + "");
                PlaySMVFragment.this.iplay_progress_vibrate.setProgress(intValue3);
                PlaySMVFragment.this.isShowMode = 0;
                if (PlaySMVFragment.this.iplay_layout_vibration.getVisibility() == 0) {
                    PlaySMVFragment playSMVFragment4 = PlaySMVFragment.this;
                    playSMVFragment4.stopAnimation(playSMVFragment4.iplay_layout_vibration);
                }
                PlaySMVFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                if (PlaySMVFragment.this.mAdapter != null) {
                    PlaySMVFragment.this.mAdapter.setClickPosition(-1);
                }
            } else if (i == 4) {
                int intValue4 = ((Integer) message.obj).intValue();
                PlaySMVFragment playSMVFragment5 = PlaySMVFragment.this;
                if (intValue4 >= 1) {
                    playSMVFragment5.isShowMode = 2;
                    if (PlaySMVFragment.this.iplay_layout_vibration.getVisibility() != 0) {
                        PlaySMVFragment playSMVFragment6 = PlaySMVFragment.this;
                        playSMVFragment6.startAnimation(playSMVFragment6.iplay_layout_vibration, PlaySMVFragment.this.iplay_layout_mode);
                    }
                    PlaySMVFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                } else {
                    playSMVFragment5.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                }
                if (PlaySMVFragment.this.mVibrationAdapter != null) {
                    int i3 = intValue4 - 1;
                    PlaySMVFragment.this.mVibrationAdapter.setClickPosition(i3);
                    if (intValue4 > 0) {
                        PlaySMVFragment.this.iplay_recycler_vibration.smoothScrollToPosition(i3);
                    }
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
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.14
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMVFragment.this.leftAction = 0;
                PlaySMVFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMVFragment.this.lastLeftClickTime;
                PlaySMVFragment playSMVFragment = PlaySMVFragment.this;
                if (j < 500) {
                    boolean z = playSMVFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMVFragment playSMVFragment2 = PlaySMVFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMVFragment2.baseActivity).exit();
                    } else if (playSMVFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMVFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMVFragment.lastLeftClickTime = currentTimeMillis;
                PlaySMVFragment.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.15
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMVFragment.this.rightAction = 0;
                PlaySMVFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMVFragment.this.lastRightClickTime;
                PlaySMVFragment playSMVFragment = PlaySMVFragment.this;
                if (j < 500) {
                    boolean z = playSMVFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMVFragment playSMVFragment2 = PlaySMVFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMVFragment2.baseActivity).exit();
                    } else if (playSMVFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMVFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMVFragment.lastRightClickTime = currentTimeMillis;
                PlaySMVFragment.this.rightAction = 1;
            }
            return true;
        }
    };

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

    /* JADX INFO: Access modifiers changed from: private */
    public SocketProductModel getBinderProduct() {
        BaseActivity baseActivity = this.baseActivity;
        return baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getBinder().getSocketProductModel() : baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getBinder().getSocketProductModel() : new SocketProductModel();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideGestureTip() {
        if (this.layout_arrow.getVisibility() == 0) {
            this.layout_arrow.setVisibility(8);
            this.arrow_both.setOnValueListener(null);
        }
        if (this.iplay_img_gesture_left.getVisibility() == 0) {
            this.iplay_img_gesture_left.clearAnimation();
            this.iplay_img_gesture_left.setVisibility(8);
        }
        if (this.iplay_img_gesture_right.getVisibility() == 0) {
            this.iplay_img_gesture_right.clearAnimation();
            this.iplay_img_gesture_right.setVisibility(8);
        }
        if (this.iplay_txt_gesture_tip.getVisibility() == 0) {
            this.iplay_txt_gesture_tip.setVisibility(8);
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
        this.iplay_recycler_mode.setLayoutManager(gridLayoutManager);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this.baseActivity, 2);
        this.mVibrationLayoutManager = gridLayoutManager2;
        this.iplay_recycler_vibration.setLayoutManager(gridLayoutManager2);
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        SocketIplayModeAdapter socketIplayModeAdapter = new SocketIplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = socketIplayModeAdapter;
        socketIplayModeAdapter.setOnItemClickListener(new SocketIplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.2
            @Override // com.hismith.hismithapp.adapter.SocketIplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                PlaySMVFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                int i2 = i + 1;
                SocketManager.getInstance().setMode(i2);
                PlaySMVFragment.this.getBinderProduct().setMode(i2);
            }
        });
        this.iplay_recycler_mode.setAdapter(this.mAdapter);
        SocketIplayModeAdapter socketIplayModeAdapter2 = new SocketIplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = socketIplayModeAdapter2;
        socketIplayModeAdapter2.setOnItemClickListener(new SocketIplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.3
            @Override // com.hismith.hismithapp.adapter.SocketIplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                PlaySMVFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
                int i2 = i + 1;
                SocketManager.getInstance().setVibrateMode(i2);
                PlaySMVFragment.this.getBinderProduct().setVibration(i2);
            }
        });
        this.iplay_recycler_vibration.setAdapter(this.mVibrationAdapter);
        if (this.mProductModel != null) {
            PlayView_Four playView_Four = this.iplay_playView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(r0.getSlide());
            }
            PlayView_Four playView_Four2 = this.iplay_playView_vibrate;
            if (playView_Four2 != null) {
                playView_Four2.setmScreenSpeed(this.mProductModel.getSlide());
            }
            initProductData();
            this.iplay_txt_speed.setText(this.mProductModel.getSpeed() + "");
            this.iplay_txt_vibrate.setText(this.mProductModel.getVspeed() + "");
            this.iplay_progress_speed.setProgress(this.mProductModel.getSpeed());
            this.iplay_progress_vibrate.setProgress(this.mProductModel.getVspeed());
            int mode = this.mProductModel.getMode();
            ImageView imageView = this.iplay_img_mode;
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
            ImageView imageView2 = this.iplay_img_vibrate;
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
        AnimUtil.translate(this.baseActivity, this.iplay_img_gesture_left);
        AnimUtil.translate(this.baseActivity, this.iplay_img_gesture_right);
        scrollModeHideImg();
        scrollVibrateHideImg();
        initSensor();
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
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                PlaySMVFragment.this.m602x743c2fc7(i, (Permission) obj);
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
        this.mBleVibrateUtil = new BleSpeedUtil(this.baseActivity);
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
        this.iplay_img_shake = (ImageView) view.findViewById(R.id.iplay_img_shake);
        this.iplay_img_mic = (ImageView) view.findViewById(R.id.iplay_img_mic);
        this.iplay_img_mode = (ImageView) view.findViewById(R.id.iplay_img_mode);
        this.iplay_img_vibrate = (ImageView) view.findViewById(R.id.iplay_img_vibrate);
        this.iplay_img_mic_vibrate = (ImageView) view.findViewById(R.id.iplay_img_mic_vibrate);
        this.iplay_img_shake_vibrate = (ImageView) view.findViewById(R.id.iplay_img_shake_vibrate);
        this.layout_arrow = (LinearLayout) view.findViewById(R.id.layout_arrow);
        this.arrow_both = (BothWayArrowView) view.findViewById(R.id.arrow_both);
        this.img_gesture = (ImageView) view.findViewById(R.id.img_gesture);
        this.txt_sliding = (TextView) view.findViewById(R.id.txt_sliding);
        this.iplay_img_close_left = (ImageView) view.findViewById(R.id.iplay_img_close_left);
        this.iplay_img_close_right = (ImageView) view.findViewById(R.id.iplay_img_close_right);
        this.iplay_img_gesture_left = (ImageView) view.findViewById(R.id.iplay_img_gesture_left);
        this.iplay_img_gesture_right = (ImageView) view.findViewById(R.id.iplay_img_gesture_right);
        this.iplay_txt_gesture_tip = (TextView) view.findViewById(R.id.iplay_txt_gesture_tip);
        this.view_touch = view.findViewById(R.id.view_touch);
        LanguageUtil.setText(this.baseActivity, this.txt_sliding, "slidingControlSpeed");
        LanguageUtil.setText(this.baseActivity, this.iplay_txt_gesture_tip, "exitDoubleGameNote");
        this.iplay_playView.setProgressBar(this.iplay_progress_speed);
        this.iplay_playView_vibrate.setProgressBar(this.iplay_progress_vibrate);
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
        this.view_touch.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                PlaySMVFragment.this.hideGestureTip();
                return false;
            }
        });
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.5
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMVFragment.this.scrollModeHideImg();
            }
        });
        this.iplay_recycler_vibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.6
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMVFragment.this.scrollVibrateHideImg();
            }
        });
        this.iplay_playView.setOnLineChangeListener(new PlayView_Four.OnLineChangeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.7
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
            public void onLineChange(int i, JsonObject jsonObject) {
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != PlaySMVFragment.this.iplay_progress_speed.getProgress() || i == 0) {
                    PlaySMVFragment.this.iplay_txt_speed.setText(i + "");
                    PlaySMVFragment.this.iplay_progress_speed.setProgress(i);
                    if (PlaySMVFragment.this.mAdapter.getClickPosition() != -1) {
                        PlaySMVFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                        PlaySMVFragment.this.mAdapter.setClickPosition(-1);
                    }
                }
                SocketManager.getInstance().setSpeed(i, jsonObject);
                PlaySMVFragment.this.getBinderProduct().setSpeed(i);
                PlaySMVFragment.this.getBinderProduct().setMode(0);
            }
        });
        this.iplay_playView_vibrate.setOnLineChangeListener(new PlayView_Four.OnLineChangeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.8
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
            public void onLineChange(int i, JsonObject jsonObject) {
                if (i <= 0) {
                    i = 0;
                } else if (i >= 100) {
                    i = 100;
                }
                if (i != PlaySMVFragment.this.iplay_progress_vibrate.getProgress() || i == 0) {
                    PlaySMVFragment.this.iplay_txt_vibrate.setText(i + "");
                    PlaySMVFragment.this.iplay_progress_vibrate.setProgress(i);
                    if (PlaySMVFragment.this.mVibrationAdapter.getClickPosition() != -1) {
                        PlaySMVFragment.this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
                        PlaySMVFragment.this.mVibrationAdapter.setClickPosition(-1);
                    }
                }
                SocketManager.getInstance().setVibrate(i, jsonObject);
                PlaySMVFragment.this.getBinderProduct().setVspeed(i);
                PlaySMVFragment.this.getBinderProduct().setVibration(0);
            }
        });
        this.arrow_both.setOnValueListener(new BothWayArrowView.OnValueListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.9
            @Override // com.hismith.hismithapp.view.BothWayArrowView.OnValueListener
            public void onValueChange(float f) {
                PlaySMVFragment.this.img_gesture.setRotation(f * (-45.0f));
            }
        });
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.10
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
                if (PlaySMVFragment.this.isMicMode) {
                    int i2 = i == 0 ? 1 : i;
                    Message message = new Message();
                    message.what = 1;
                    message.obj = Integer.valueOf(i2);
                    PlaySMVFragment.this.mHandler.sendMessage(message);
                    SocketManager.getInstance().setSpeed(i2);
                    PlaySMVFragment.this.getBinderProduct().setSpeed(i2);
                    PlaySMVFragment.this.getBinderProduct().setMode(0);
                }
                if (PlaySMVFragment.this.isMicVibrateMode) {
                    PlaySMVFragment.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.10.1
                        @Override // java.lang.Runnable
                        public void run() {
                            Message message2 = new Message();
                            message2.what = 3;
                            message2.obj = Integer.valueOf(i);
                            PlaySMVFragment.this.mHandler.sendMessage(message2);
                            SocketManager.getInstance().setVibrate(i);
                            PlaySMVFragment.this.getBinderProduct().setVspeed(i);
                            PlaySMVFragment.this.getBinderProduct().setVibration(0);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.11
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMVFragment.this.inAnim = false;
                if (i != PlaySMVFragment.this.isShowMode) {
                    PlaySMVFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMVFragment.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.12
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMVFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMVFragment.this.inAnim = true;
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
            if (this.recorderVolumeManager.isRecording()) {
                return;
            }
            this.recorderVolumeManager.startRecording();
            return;
        }
        imageView.setImageResource(R.mipmap.ic_iplay_mic);
        if (!this.isMicVibrateMode) {
            this.recorderVolumeManager.stopRecording();
        }
        SocketManager.getInstance().setSpeed(0);
        getBinderProduct().setSpeed(0);
        getBinderProduct().setMode(0);
        Message message = new Message();
        message.what = 1;
        message.obj = 0;
        this.mHandler.sendMessage(message);
    }

    private void startMicVibrateClick() {
        boolean z = this.isMicVibrateMode;
        this.isMicVibrateMode = !z;
        ImageView imageView = this.iplay_img_mic_vibrate;
        if (!z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic_select);
            if (this.recorderVolumeManager.isRecording()) {
                return;
            }
            this.recorderVolumeManager.startRecording();
            return;
        }
        imageView.setImageResource(R.mipmap.ic_iplay_mic);
        if (!this.isMicMode) {
            this.recorderVolumeManager.stopRecording();
        }
        SocketManager.getInstance().setVibrate(0);
        getBinderProduct().setVspeed(0);
        getBinderProduct().setVibration(0);
        Message message = new Message();
        message.what = 3;
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.13
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMVFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != PlaySMVFragment.this.isShowMode) {
                    PlaySMVFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMVFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
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
        if (this.mProductModel.getModesUrl().getVibrations() != null) {
            this.mVibrationModes.addAll(Arrays.asList(this.mProductModel.getModesUrl().getVibrations()));
        }
        this.mVibrationAdapter.notifyDataSetChanged();
        scrollVibrateHideImg();
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-fragment-PlaySMVFragment, reason: not valid java name */
    /* synthetic */ void m602x743c2fc7(int i, Permission permission) throws Exception {
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
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.18
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + PlaySMVFragment.this.baseActivity.getPackageName()));
                    PlaySMVFragment.this.startActivity(intent);
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
        if (id == R.id.iplay_img_mic) {
            initRecordPermission(1);
            return;
        }
        if (id == R.id.iplay_img_mic_vibrate) {
            initRecordPermission(2);
            return;
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
        if (id == R.id.iplay_img_close_mode) {
            int progress = this.iplay_progress_speed.getProgress();
            SocketManager.getInstance().setSpeed(progress);
            getBinderProduct().setSpeed(progress);
            getBinderProduct().setMode(0);
            Message message = new Message();
            message.what = 1;
            message.obj = Integer.valueOf(progress);
            this.mHandler.sendMessage(message);
            return;
        }
        if (id == R.id.iplay_img_close_vibration) {
            int progress2 = this.iplay_progress_vibrate.getProgress();
            SocketManager.getInstance().setVibrate(progress2);
            getBinderProduct().setVspeed(progress2);
            getBinderProduct().setVibration(0);
            Message message2 = new Message();
            message2.what = 3;
            message2.obj = Integer.valueOf(progress2);
            this.mHandler.sendMessage(message2);
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
        View inflate = layoutInflater.inflate(R.layout.fragment_iplay_smv, viewGroup, false);
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
            int shake = (socketProductModel == null || socketProductModel.getShake() <= 0) ? 100 : this.mProductModel.getShake();
            this.mBleSpeedUtil.shakeSpeed(f, f2, f3, shake, new BleSpeedUtil.OnShakeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.16
                @Override // com.hismith.hismithapp.utils.BleSpeedUtil.OnShakeListener
                public void onShake(int i) {
                    Message message = new Message();
                    message.what = 1;
                    message.obj = Integer.valueOf(i);
                    PlaySMVFragment.this.mHandler.sendMessage(message);
                    SocketManager.getInstance().setSpeed(i);
                    PlaySMVFragment.this.getBinderProduct().setSpeed(i);
                    PlaySMVFragment.this.getBinderProduct().setMode(0);
                }
            });
            this.mBleVibrateUtil.shakeSpeed(f, f2, f3, shake, new BleSpeedUtil.OnShakeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMVFragment.17
                @Override // com.hismith.hismithapp.utils.BleSpeedUtil.OnShakeListener
                public void onShake(int i) {
                    Message message = new Message();
                    message.what = 3;
                    message.obj = Integer.valueOf(i);
                    PlaySMVFragment.this.mHandler.sendMessage(message);
                    SocketManager.getInstance().setVibrate(i);
                    PlaySMVFragment.this.getBinderProduct().setVspeed(i);
                    PlaySMVFragment.this.getBinderProduct().setVibration(0);
                }
            });
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
