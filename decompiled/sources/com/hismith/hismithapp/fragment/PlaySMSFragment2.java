package com.hismith.hismithapp.fragment;

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
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.ConstraintSet;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentTransaction;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.gms.common.util.ArrayUtils;
import com.google.gson.Gson;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.SocketIplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.webfragment.PlaySMMyFragment;
import com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment;
import com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment;
import com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment;
import com.hismith.hismithapp.fragment.webfragment.PlaySMVYouFragment;
import com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.BleSpeedUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMSFragment2 extends SocketPlayFragment implements View.OnClickListener, SensorEventListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private final int WHAT_MODE;
    private final int WHAT_VIBRATION_MODE;
    private SocketPlayFragment YouFragment;
    private int animationDuration;
    private BaseActivity baseActivity;
    private ImageView img_close_left;
    private ImageView img_close_mode;
    private ImageView img_close_right;
    private ImageView img_close_vibrate_mode;
    private ImageView img_speed_mic;
    private ImageView img_speed_mode;
    private ImageView img_speed_shake;
    private ImageView img_vibrate_mic;
    private ImageView img_vibrate_mode;
    private ImageView img_vibrate_shake;
    private boolean inAnim;
    private boolean isMicMode;
    private boolean isMicVibrateMode;
    private int isShowMode;
    private long lastLeftClickTime;
    private long lastRightClickTime;
    private FrameLayout layout_arrow_down;
    private FrameLayout layout_arrow_up;
    private FrameLayout layout_frame_my;
    private FrameLayout layout_frame_you;
    private ConstraintLayout layout_mode_list;
    private ConstraintLayout layout_root;
    private FrameLayout layout_vibrate_arrow_down;
    private FrameLayout layout_vibrate_arrow_up;
    private ConstraintLayout layout_vibrate_mode_list;
    private int leftAction;
    private View.OnTouchListener leftOnTouch;
    private SocketIplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private BleSpeedUtil mBleSpeedUtil;
    private BleSpeedUtil mBleVibrateUtil;
    private SocketPlayFragment mFragment;
    private Gson mGson;
    private Handler mHandler;
    private GridLayoutManager mLayoutManager;
    private List<String> mModes;
    private SocketProductModel mProductModel;
    private BroadcastReceiver mReceiver;
    private int mSafeMax;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private int mStartPositionSafeMax;
    private int mStopPositionSafeMax;
    private SocketIplayModeAdapter mVibrateAdapter;
    private GridLayoutManager mVibrateLayoutManager;
    private List<String> mVibrateModes;
    private int mVibrateSafeMax;
    private RecorderVolumeManager recorderVolumeManager;
    private RecyclerView recycler_mode;
    private RecyclerView recycler_vibrate_mode;
    private int rightAction;
    private View.OnTouchListener rightOnTouch;

    public PlaySMSFragment2() {
        this.isMicMode = false;
        this.isMicVibrateMode = false;
        this.leftAction = 1;
        this.rightAction = 1;
        this.mSafeMax = 100;
        this.mVibrateSafeMax = 100;
        this.mStartPositionSafeMax = 0;
        this.mStopPositionSafeMax = 100;
        this.WHAT_MODE = 2;
        this.WHAT_VIBRATION_MODE = 4;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int i = message.what;
                if (i == 2) {
                    int intValue = ((Integer) message.obj).intValue();
                    LogUtil.e("Handler mode:" + intValue);
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (intValue <= 0) {
                        playSMSFragment2.img_speed_mode.setImageResource(R.mipmap.ic_iplay_mode);
                    } else {
                        playSMSFragment2.img_speed_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                    }
                    if (PlaySMSFragment2.this.mAdapter != null) {
                        int i2 = intValue - 1;
                        PlaySMSFragment2.this.mAdapter.setClickPosition(i2);
                        if (intValue > 0) {
                            PlaySMSFragment2.this.recycler_mode.smoothScrollToPosition(i2);
                        }
                    }
                    PlaySMSFragment2.this.getBinderProduct().setMode(intValue);
                    return false;
                }
                if (i != 4) {
                    return false;
                }
                int intValue2 = ((Integer) message.obj).intValue();
                PlaySMSFragment2 playSMSFragment22 = PlaySMSFragment2.this;
                if (intValue2 >= 1) {
                    playSMSFragment22.img_vibrate_mode.setImageResource(R.mipmap.ic_iplay_vibration_select);
                } else {
                    playSMSFragment22.img_vibrate_mode.setImageResource(R.mipmap.ic_iplay_vibration);
                }
                if (PlaySMSFragment2.this.mVibrateAdapter != null) {
                    int i3 = intValue2 - 1;
                    PlaySMSFragment2.this.mVibrateAdapter.setClickPosition(i3);
                    if (intValue2 > 0) {
                        PlaySMSFragment2.this.recycler_mode.smoothScrollToPosition(i3);
                    }
                }
                PlaySMSFragment2.this.getBinderProduct().setVibration(intValue2);
                return false;
            }
        });
        this.lastLeftClickTime = 0L;
        this.lastRightClickTime = 0L;
        this.leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.7
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlaySMSFragment2.this.leftAction = 0;
                    PlaySMSFragment2.this.exitIPlay();
                } else if (action == 1) {
                    long currentTimeMillis = System.currentTimeMillis();
                    long j = currentTimeMillis - PlaySMSFragment2.this.lastLeftClickTime;
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (j < 500) {
                        playSMSFragment2.exit();
                        return true;
                    }
                    playSMSFragment2.lastLeftClickTime = currentTimeMillis;
                    PlaySMSFragment2.this.leftAction = 1;
                }
                LogUtil.e("left action:" + motionEvent.getAction());
                return true;
            }
        };
        this.rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.8
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlaySMSFragment2.this.rightAction = 0;
                    PlaySMSFragment2.this.exitIPlay();
                } else if (action == 1) {
                    long currentTimeMillis = System.currentTimeMillis();
                    long j = currentTimeMillis - PlaySMSFragment2.this.lastRightClickTime;
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (j < 500) {
                        playSMSFragment2.exit();
                        return true;
                    }
                    playSMSFragment2.lastRightClickTime = currentTimeMillis;
                    PlaySMSFragment2.this.rightAction = 1;
                }
                LogUtil.e("right action:" + motionEvent.getAction());
                return true;
            }
        };
        this.inAnim = false;
        this.isShowMode = 0;
        this.animationDuration = 300;
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.13
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();
                if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                    PlaySMSFragment2.this.initFragment();
                    if (PlaySMSFragment2.this.mFragment == null || !PlaySMSFragment2.this.mFragment.isAdded()) {
                        return;
                    }
                    PlaySMSFragment2.this.mFragment.initProductData();
                    return;
                }
                if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                    String stringExtra = intent.getStringExtra(Constans.BleType);
                    String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                    String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                    LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                    if (PlaySMSFragment2.this.mFragment == null || !PlaySMSFragment2.this.mFragment.isAdded()) {
                        return;
                    }
                    if ("AA".equals(stringExtra)) {
                        PlaySMSFragment2.this.mFragment.initBleAAData(stringExtra2, stringExtra3);
                    }
                    if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                        PlaySMSFragment2.this.mFragment.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                    }
                }
            }
        };
    }

    public PlaySMSFragment2(int i) {
        this.isMicMode = false;
        this.isMicVibrateMode = false;
        this.leftAction = 1;
        this.rightAction = 1;
        this.mSafeMax = 100;
        this.mVibrateSafeMax = 100;
        this.mStartPositionSafeMax = 0;
        this.mStopPositionSafeMax = 100;
        this.WHAT_MODE = 2;
        this.WHAT_VIBRATION_MODE = 4;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int i2 = message.what;
                if (i2 == 2) {
                    int intValue = ((Integer) message.obj).intValue();
                    LogUtil.e("Handler mode:" + intValue);
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (intValue <= 0) {
                        playSMSFragment2.img_speed_mode.setImageResource(R.mipmap.ic_iplay_mode);
                    } else {
                        playSMSFragment2.img_speed_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                    }
                    if (PlaySMSFragment2.this.mAdapter != null) {
                        int i22 = intValue - 1;
                        PlaySMSFragment2.this.mAdapter.setClickPosition(i22);
                        if (intValue > 0) {
                            PlaySMSFragment2.this.recycler_mode.smoothScrollToPosition(i22);
                        }
                    }
                    PlaySMSFragment2.this.getBinderProduct().setMode(intValue);
                    return false;
                }
                if (i2 != 4) {
                    return false;
                }
                int intValue2 = ((Integer) message.obj).intValue();
                PlaySMSFragment2 playSMSFragment22 = PlaySMSFragment2.this;
                if (intValue2 >= 1) {
                    playSMSFragment22.img_vibrate_mode.setImageResource(R.mipmap.ic_iplay_vibration_select);
                } else {
                    playSMSFragment22.img_vibrate_mode.setImageResource(R.mipmap.ic_iplay_vibration);
                }
                if (PlaySMSFragment2.this.mVibrateAdapter != null) {
                    int i3 = intValue2 - 1;
                    PlaySMSFragment2.this.mVibrateAdapter.setClickPosition(i3);
                    if (intValue2 > 0) {
                        PlaySMSFragment2.this.recycler_mode.smoothScrollToPosition(i3);
                    }
                }
                PlaySMSFragment2.this.getBinderProduct().setVibration(intValue2);
                return false;
            }
        });
        this.lastLeftClickTime = 0L;
        this.lastRightClickTime = 0L;
        this.leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.7
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlaySMSFragment2.this.leftAction = 0;
                    PlaySMSFragment2.this.exitIPlay();
                } else if (action == 1) {
                    long currentTimeMillis = System.currentTimeMillis();
                    long j = currentTimeMillis - PlaySMSFragment2.this.lastLeftClickTime;
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (j < 500) {
                        playSMSFragment2.exit();
                        return true;
                    }
                    playSMSFragment2.lastLeftClickTime = currentTimeMillis;
                    PlaySMSFragment2.this.leftAction = 1;
                }
                LogUtil.e("left action:" + motionEvent.getAction());
                return true;
            }
        };
        this.rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.8
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlaySMSFragment2.this.rightAction = 0;
                    PlaySMSFragment2.this.exitIPlay();
                } else if (action == 1) {
                    long currentTimeMillis = System.currentTimeMillis();
                    long j = currentTimeMillis - PlaySMSFragment2.this.lastRightClickTime;
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (j < 500) {
                        playSMSFragment2.exit();
                        return true;
                    }
                    playSMSFragment2.lastRightClickTime = currentTimeMillis;
                    PlaySMSFragment2.this.rightAction = 1;
                }
                LogUtil.e("right action:" + motionEvent.getAction());
                return true;
            }
        };
        this.inAnim = false;
        this.isShowMode = 0;
        this.animationDuration = 300;
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.13
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();
                if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                    PlaySMSFragment2.this.initFragment();
                    if (PlaySMSFragment2.this.mFragment == null || !PlaySMSFragment2.this.mFragment.isAdded()) {
                        return;
                    }
                    PlaySMSFragment2.this.mFragment.initProductData();
                    return;
                }
                if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                    String stringExtra = intent.getStringExtra(Constans.BleType);
                    String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                    String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                    LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                    if (PlaySMSFragment2.this.mFragment == null || !PlaySMSFragment2.this.mFragment.isAdded()) {
                        return;
                    }
                    if ("AA".equals(stringExtra)) {
                        PlaySMSFragment2.this.mFragment.initBleAAData(stringExtra2, stringExtra3);
                    }
                    if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                        PlaySMSFragment2.this.mFragment.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                    }
                }
            }
        };
        this.mSafeMax = i;
    }

    public PlaySMSFragment2(int i, int i2) {
        this.isMicMode = false;
        this.isMicVibrateMode = false;
        this.leftAction = 1;
        this.rightAction = 1;
        this.mSafeMax = 100;
        this.mVibrateSafeMax = 100;
        this.mStartPositionSafeMax = 0;
        this.mStopPositionSafeMax = 100;
        this.WHAT_MODE = 2;
        this.WHAT_VIBRATION_MODE = 4;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int i22 = message.what;
                if (i22 == 2) {
                    int intValue = ((Integer) message.obj).intValue();
                    LogUtil.e("Handler mode:" + intValue);
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (intValue <= 0) {
                        playSMSFragment2.img_speed_mode.setImageResource(R.mipmap.ic_iplay_mode);
                    } else {
                        playSMSFragment2.img_speed_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                    }
                    if (PlaySMSFragment2.this.mAdapter != null) {
                        int i222 = intValue - 1;
                        PlaySMSFragment2.this.mAdapter.setClickPosition(i222);
                        if (intValue > 0) {
                            PlaySMSFragment2.this.recycler_mode.smoothScrollToPosition(i222);
                        }
                    }
                    PlaySMSFragment2.this.getBinderProduct().setMode(intValue);
                    return false;
                }
                if (i22 != 4) {
                    return false;
                }
                int intValue2 = ((Integer) message.obj).intValue();
                PlaySMSFragment2 playSMSFragment22 = PlaySMSFragment2.this;
                if (intValue2 >= 1) {
                    playSMSFragment22.img_vibrate_mode.setImageResource(R.mipmap.ic_iplay_vibration_select);
                } else {
                    playSMSFragment22.img_vibrate_mode.setImageResource(R.mipmap.ic_iplay_vibration);
                }
                if (PlaySMSFragment2.this.mVibrateAdapter != null) {
                    int i3 = intValue2 - 1;
                    PlaySMSFragment2.this.mVibrateAdapter.setClickPosition(i3);
                    if (intValue2 > 0) {
                        PlaySMSFragment2.this.recycler_mode.smoothScrollToPosition(i3);
                    }
                }
                PlaySMSFragment2.this.getBinderProduct().setVibration(intValue2);
                return false;
            }
        });
        this.lastLeftClickTime = 0L;
        this.lastRightClickTime = 0L;
        this.leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.7
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlaySMSFragment2.this.leftAction = 0;
                    PlaySMSFragment2.this.exitIPlay();
                } else if (action == 1) {
                    long currentTimeMillis = System.currentTimeMillis();
                    long j = currentTimeMillis - PlaySMSFragment2.this.lastLeftClickTime;
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (j < 500) {
                        playSMSFragment2.exit();
                        return true;
                    }
                    playSMSFragment2.lastLeftClickTime = currentTimeMillis;
                    PlaySMSFragment2.this.leftAction = 1;
                }
                LogUtil.e("left action:" + motionEvent.getAction());
                return true;
            }
        };
        this.rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.8
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlaySMSFragment2.this.rightAction = 0;
                    PlaySMSFragment2.this.exitIPlay();
                } else if (action == 1) {
                    long currentTimeMillis = System.currentTimeMillis();
                    long j = currentTimeMillis - PlaySMSFragment2.this.lastRightClickTime;
                    PlaySMSFragment2 playSMSFragment2 = PlaySMSFragment2.this;
                    if (j < 500) {
                        playSMSFragment2.exit();
                        return true;
                    }
                    playSMSFragment2.lastRightClickTime = currentTimeMillis;
                    PlaySMSFragment2.this.rightAction = 1;
                }
                LogUtil.e("right action:" + motionEvent.getAction());
                return true;
            }
        };
        this.inAnim = false;
        this.isShowMode = 0;
        this.animationDuration = 300;
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.13
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();
                if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                    PlaySMSFragment2.this.initFragment();
                    if (PlaySMSFragment2.this.mFragment == null || !PlaySMSFragment2.this.mFragment.isAdded()) {
                        return;
                    }
                    PlaySMSFragment2.this.mFragment.initProductData();
                    return;
                }
                if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                    String stringExtra = intent.getStringExtra(Constans.BleType);
                    String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                    String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                    LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                    if (PlaySMSFragment2.this.mFragment == null || !PlaySMSFragment2.this.mFragment.isAdded()) {
                        return;
                    }
                    if ("AA".equals(stringExtra)) {
                        PlaySMSFragment2.this.mFragment.initBleAAData(stringExtra2, stringExtra3);
                    }
                    if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                        PlaySMSFragment2.this.mFragment.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                    }
                }
            }
        };
        this.mSafeMax = i;
        this.mVibrateSafeMax = i2;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exit() {
        BaseActivity baseActivity = this.baseActivity;
        if (baseActivity instanceof ControlSponsorActivity2) {
            ((ControlSponsorActivity2) baseActivity).exit();
        } else if (baseActivity instanceof ControlAcceptActivity2) {
            ((ControlAcceptActivity2) baseActivity).exit();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exitIPlay() {
        if (this.leftAction == 0 && this.rightAction == 0) {
            exit();
        }
        if (this.leftAction == 1) {
            LogUtil.e("onTouch 左抖动");
            AnimUtil.shakeAnimation(this.baseActivity, this.img_close_left);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            LogUtil.e("onTouch 右抖动");
            AnimUtil.shakeAnimation(this.baseActivity, this.img_close_right);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public SocketProductModel getBinderProduct() {
        BaseActivity baseActivity = this.baseActivity;
        SocketProductModel socketProductModel = baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getBinder().getSocketProductModel() : baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getBinder().getSocketProductModel() : null;
        return socketProductModel == null ? new SocketProductModel() : socketProductModel;
    }

    private void init() {
        this.mGson = new Gson();
        Bundle arguments = getArguments();
        if (arguments != null) {
            this.mProductModel = (SocketProductModel) arguments.getSerializable("ProductModel");
        }
        this.recorderVolumeManager = new RecorderVolumeManager();
        initSensor();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.recycler_mode.setLayoutManager(gridLayoutManager);
        this.mModes = new ArrayList();
        SocketIplayModeAdapter socketIplayModeAdapter = new SocketIplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = socketIplayModeAdapter;
        socketIplayModeAdapter.setOnItemClickListener(new SocketIplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.2
            @Override // com.hismith.hismithapp.adapter.SocketIplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                int i2 = i + 1;
                SocketManager.getInstance().setMode(i2);
                PlaySMSFragment2.this.refreshSpeedModeUI(i2);
            }
        });
        this.recycler_mode.setAdapter(this.mAdapter);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this.baseActivity, 2);
        this.mVibrateLayoutManager = gridLayoutManager2;
        this.recycler_vibrate_mode.setLayoutManager(gridLayoutManager2);
        this.mVibrateModes = new ArrayList();
        SocketIplayModeAdapter socketIplayModeAdapter2 = new SocketIplayModeAdapter(this.baseActivity, this.mVibrateModes);
        this.mVibrateAdapter = socketIplayModeAdapter2;
        socketIplayModeAdapter2.setOnItemClickListener(new SocketIplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.3
            @Override // com.hismith.hismithapp.adapter.SocketIplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                int i2 = i + 1;
                SocketManager.getInstance().setVibrateMode(i2);
                PlaySMSFragment2.this.refreshVibrateModeUI(i2);
            }
        });
        this.recycler_vibrate_mode.setAdapter(this.mVibrateAdapter);
        initFragment();
        registerReceiver();
        initProduct(this.mProductModel);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initFragment() {
        SocketPlayFragment socketPlayFragment;
        FragmentTransaction beginTransaction = getChildFragmentManager().beginTransaction();
        SocketPlayFragment playSMSMyFragment = ProductUtil.isSMSType() ? new PlaySMSMyFragment(this.mSafeMax) : ProductUtil.isSMVType() ? new PlaySMVMyFragment(this.mSafeMax, this.mVibrateSafeMax) : (ProductUtil.isSM2Type() || ProductUtil.isSMType()) ? new PlaySMMyFragment(this.mSafeMax) : null;
        if (playSMSMyFragment != null) {
            this.layout_frame_my.setVisibility(0);
            SocketPlayFragment socketPlayFragment2 = this.mFragment;
            if (socketPlayFragment2 == null || socketPlayFragment2.getClass() != playSMSMyFragment.getClass()) {
                this.mFragment = playSMSMyFragment;
                beginTransaction.replace(R.id.layout_frame_my, this.mFragment);
            }
        } else {
            SocketPlayFragment socketPlayFragment3 = this.mFragment;
            if (socketPlayFragment3 != null) {
                beginTransaction.remove(socketPlayFragment3);
                this.mFragment = null;
            }
            this.layout_frame_my.setVisibility(8);
        }
        LogUtil.e("SocketManager device type:" + this.mProductModel.getType());
        SocketProductModel socketProductModel = this.mProductModel;
        if (socketProductModel == null || TextUtils.isEmpty(socketProductModel.getType())) {
            socketPlayFragment = null;
        } else if (this.mProductModel.getType().equals(Constans.UI_DEVICE_TYPE_CC_SMS)) {
            socketPlayFragment = new PlaySMSYouFragment();
        } else if (this.mProductModel.getType().equals(Constans.UI_DEVICE_TYPE_CC_SMV)) {
            LogUtil.e("SocketManager device ui:PlaySMVYouFragment");
            socketPlayFragment = new PlaySMVYouFragment();
        } else {
            socketPlayFragment = new PlaySMYouFragment();
        }
        if (socketPlayFragment != null) {
            this.layout_frame_you.setVisibility(0);
            SocketPlayFragment socketPlayFragment4 = this.YouFragment;
            if (socketPlayFragment4 == null || socketPlayFragment4.getClass() != socketPlayFragment.getClass()) {
                this.YouFragment = socketPlayFragment;
                Bundle bundle = new Bundle();
                bundle.putSerializable("ProductModel", this.mProductModel);
                this.YouFragment.setArguments(bundle);
                beginTransaction.replace(R.id.layout_frame_you, this.YouFragment);
            }
        } else {
            SocketPlayFragment socketPlayFragment5 = this.YouFragment;
            if (socketPlayFragment5 != null) {
                beginTransaction.remove(socketPlayFragment5);
                this.YouFragment = null;
            }
            this.layout_frame_you.setVisibility(8);
        }
        if (MyApplication.getApplication().getmProductModel() == null || this.mProductModel == null || !"01device".equals(MyApplication.getApplication().getmProductModel().getSvgName()) || !"01device".equals(this.mProductModel.getSvgName())) {
            ConstraintSet constraintSet = new ConstraintSet();
            constraintSet.clone(this.layout_root);
            constraintSet.connect(R.id.line_1, 3, R.id.img_vibrate_mode, 4);
            constraintSet.clear(R.id.line_1, 4);
            constraintSet.applyTo(this.layout_root);
        } else {
            ConstraintSet constraintSet2 = new ConstraintSet();
            constraintSet2.clone(this.layout_root);
            constraintSet2.connect(R.id.line_1, 3, R.id.layout_frame_my, 3);
            constraintSet2.connect(R.id.line_1, 4, R.id.layout_frame_my, 4);
            constraintSet2.setVerticalBias(R.id.line_1, 0.73f);
            constraintSet2.applyTo(this.layout_root);
        }
        beginTransaction.commitAllowingStateLoss();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initModeShow() {
        int i = this.isShowMode;
        if (i != 0) {
            if (i == 1) {
                startAnimation(this.layout_mode_list, this.layout_vibrate_mode_list);
                return;
            } else {
                if (i == 2) {
                    startAnimation(this.layout_vibrate_mode_list, this.layout_mode_list);
                    return;
                }
                return;
            }
        }
        ConstraintLayout constraintLayout = this.layout_mode_list;
        if (constraintLayout != null && constraintLayout.getVisibility() == 0) {
            stopAnimation(this.layout_mode_list);
        }
        ConstraintLayout constraintLayout2 = this.layout_vibrate_mode_list;
        if (constraintLayout2 == null || constraintLayout2.getVisibility() != 0) {
            return;
        }
        stopAnimation(this.layout_vibrate_mode_list);
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
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                PlaySMSFragment2.this.m601x366d75ee(i, (Permission) obj);
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

    private void initSocketData(String str) {
        try {
            SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel != null && socketSpeedModel.getOnlyRead() == 1) {
                if ("setSpeed".equals(socketSpeedModel.getAction())) {
                    if (socketSpeedModel.isMotor1()) {
                        refreshSpeedModeUI(0);
                        return;
                    } else {
                        if (socketSpeedModel.isMotor2()) {
                            refreshVibrateModeUI(0);
                            return;
                        }
                        return;
                    }
                }
                return;
            }
            if (socketSpeedModel == null || !"switchMode".equals(socketSpeedModel.getAction())) {
                return;
            }
            if (socketSpeedModel.getStatus() != 1) {
                if (socketSpeedModel.getStatus() == 0) {
                    this.img_speed_mode.setVisibility(4);
                    this.img_vibrate_mode.setVisibility(4);
                    return;
                }
                return;
            }
            SocketProductModel socketProductModel = this.mProductModel;
            if (socketProductModel == null || TextUtils.isEmpty(socketProductModel.getType()) || this.mProductModel.getModesUrl() == null) {
                return;
            }
            if (this.mProductModel.getModesUrl().getModes() != null) {
                this.img_speed_mode.setVisibility(0);
            }
            if (this.mProductModel.getModesUrl().getVibrations() != null) {
                this.img_vibrate_mode.setVisibility(0);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initView(View view) {
        this.layout_root = (ConstraintLayout) view.findViewById(R.id.layout_root);
        this.layout_frame_my = (FrameLayout) view.findViewById(R.id.layout_frame_my);
        this.layout_frame_you = (FrameLayout) view.findViewById(R.id.layout_frame_you);
        this.img_close_left = (ImageView) view.findViewById(R.id.img_close_left);
        this.img_close_right = (ImageView) view.findViewById(R.id.img_close_right);
        this.img_speed_mode = (ImageView) view.findViewById(R.id.img_speed_mode);
        this.img_speed_mic = (ImageView) view.findViewById(R.id.img_speed_mic);
        this.img_speed_shake = (ImageView) view.findViewById(R.id.img_speed_shake);
        this.img_vibrate_mode = (ImageView) view.findViewById(R.id.img_vibrate_mode);
        this.img_vibrate_mic = (ImageView) view.findViewById(R.id.img_vibrate_mic);
        this.img_vibrate_shake = (ImageView) view.findViewById(R.id.img_vibrate_shake);
        this.layout_mode_list = (ConstraintLayout) view.findViewById(R.id.layout_mode_list);
        this.recycler_mode = (RecyclerView) view.findViewById(R.id.recycler_mode);
        this.layout_arrow_up = (FrameLayout) view.findViewById(R.id.layout_arrow_up);
        this.layout_arrow_down = (FrameLayout) view.findViewById(R.id.layout_arrow_down);
        this.img_close_mode = (ImageView) view.findViewById(R.id.img_close_mode);
        this.layout_vibrate_mode_list = (ConstraintLayout) view.findViewById(R.id.layout_vibrate_mode_list);
        this.recycler_vibrate_mode = (RecyclerView) view.findViewById(R.id.recycler_vibrate_mode);
        this.layout_vibrate_arrow_up = (FrameLayout) view.findViewById(R.id.layout_vibrate_arrow_up);
        this.layout_vibrate_arrow_down = (FrameLayout) view.findViewById(R.id.layout_vibrate_arrow_down);
        this.img_close_vibrate_mode = (ImageView) view.findViewById(R.id.img_close_vibrate_mode);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        if (this.recycler_mode.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.layout_arrow_down.getVisibility() == 4) {
                this.layout_arrow_down.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.layout_arrow_down.setVisibility(4);
        }
        if (!this.recycler_mode.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.layout_arrow_up.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.layout_arrow_up.getVisibility() == 4) {
                this.layout_arrow_up.setVisibility(0);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollVibrateHideImg() {
        if (this.recycler_vibrate_mode.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.layout_vibrate_arrow_down.getVisibility() == 4) {
                this.layout_vibrate_arrow_down.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.layout_vibrate_arrow_down.setVisibility(4);
        }
        if (!this.recycler_vibrate_mode.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.layout_vibrate_arrow_up.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.layout_vibrate_arrow_up.getVisibility() == 4) {
                this.layout_vibrate_arrow_up.setVisibility(0);
            }
        }
    }

    private void setListener() {
        this.img_close_left.setOnTouchListener(this.leftOnTouch);
        this.img_close_right.setOnTouchListener(this.rightOnTouch);
        this.img_speed_mode.setOnClickListener(this);
        this.img_vibrate_mode.setOnClickListener(this);
        this.img_close_mode.setOnClickListener(this);
        this.img_close_vibrate_mode.setOnClickListener(this);
        this.recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.4
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMSFragment2.this.scrollModeHideImg();
            }
        });
        this.recycler_vibrate_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.5
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMSFragment2.this.scrollVibrateHideImg();
            }
        });
        this.img_speed_mic.setOnClickListener(this);
        this.img_speed_shake.setOnClickListener(this);
        this.img_vibrate_mic.setOnClickListener(this);
        this.img_vibrate_shake.setOnClickListener(this);
        this.layout_arrow_up.setOnClickListener(this);
        this.layout_arrow_down.setOnClickListener(this);
        this.layout_vibrate_arrow_up.setOnClickListener(this);
        this.layout_vibrate_arrow_down.setOnClickListener(this);
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.6
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
                if (PlaySMSFragment2.this.isMicMode) {
                    SocketManager.getInstance().setSpeed(i);
                    PlaySMSFragment2.this.getBinderProduct().setSpeed(i);
                    PlaySMSFragment2.this.refreshSpeedModeUI(0);
                }
                if (PlaySMSFragment2.this.isMicVibrateMode) {
                    PlaySMSFragment2.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.6.1
                        @Override // java.lang.Runnable
                        public void run() {
                            SocketManager.getInstance().setVibrate(i);
                            PlaySMSFragment2.this.getBinderProduct().setVspeed(i);
                            PlaySMSFragment2.this.refreshVibrateModeUI(0);
                        }
                    }, 50L);
                }
            }
        });
    }

    private void startAnimation(final ViewGroup viewGroup, final ViewGroup viewGroup2) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(viewGroup, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(viewGroup, "scaleY", 0.0f, 1.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.10
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMSFragment2.this.inAnim = false;
                if (i != PlaySMSFragment2.this.isShowMode) {
                    PlaySMSFragment2.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMSFragment2.this.inAnim = true;
                viewGroup.setVisibility(0);
            }
        });
        if (viewGroup2 == null || viewGroup2.getVisibility() != 0) {
            if (viewGroup.getVisibility() != 0) {
                AnimatorSet animatorSet = new AnimatorSet();
                this.mAnimatorSet = animatorSet;
                animatorSet.playTogether(ofFloat, ofFloat2);
                this.mAnimatorSet.start();
                return;
            }
            return;
        }
        ObjectAnimator ofFloat3 = ObjectAnimator.ofFloat(viewGroup2, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat4 = ObjectAnimator.ofFloat(viewGroup2, "scaleY", 1.0f, 0.0f);
        ofFloat3.setDuration(this.animationDuration);
        ofFloat4.setDuration(this.animationDuration);
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.11
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMSFragment2.this.inAnim = false;
                viewGroup2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMSFragment2.this.inAnim = true;
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
        ImageView imageView = this.img_speed_mic;
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
        ImageView imageView = this.img_vibrate_mic;
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

    private void stopAnimation(final ViewGroup viewGroup) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(viewGroup, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(viewGroup, "scaleY", 1.0f, 0.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.12
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMSFragment2.this.inAnim = false;
                viewGroup.setVisibility(8);
                if (i != PlaySMSFragment2.this.isShowMode) {
                    PlaySMSFragment2.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMSFragment2.this.inAnim = true;
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

    public void initProduct(SocketProductModel socketProductModel) {
        if (socketProductModel != null) {
            if (socketProductModel.getShake() > 0) {
                this.img_speed_shake.setVisibility(0);
                boolean equals = socketProductModel.getType().equals(Constans.UI_DEVICE_TYPE_CC_SMV);
                ImageView imageView = this.img_vibrate_shake;
                if (equals) {
                    imageView.setVisibility(0);
                } else {
                    imageView.setVisibility(8);
                }
            } else {
                this.mBleSpeedUtil.setShake(false);
                this.mBleVibrateUtil.setShake(false);
                this.img_speed_shake.setVisibility(8);
                this.img_vibrate_shake.setVisibility(8);
            }
        }
        if (socketProductModel == null || TextUtils.isEmpty(socketProductModel.getType()) || socketProductModel.getModesUrl() == null) {
            this.img_speed_mode.setVisibility(4);
            this.img_speed_mic.setVisibility(8);
            this.img_vibrate_mode.setVisibility(4);
            this.img_vibrate_mic.setVisibility(8);
            return;
        }
        if (socketProductModel.getModesUrl().getModes() != null) {
            if (socketProductModel.getModeStatus() == 1) {
                this.img_speed_mode.setVisibility(0);
            } else if (socketProductModel.getModeStatus() == 0) {
                this.img_speed_mode.setVisibility(4);
            }
            this.img_speed_mic.setVisibility(0);
            this.mModes.clear();
            this.mModes.addAll(ArrayUtils.toArrayList(socketProductModel.getModesUrl().getModes()));
            this.mAdapter.notifyDataSetChanged();
            scrollModeHideImg();
        } else {
            this.img_speed_mode.setVisibility(4);
            this.img_speed_mic.setVisibility(8);
        }
        if (socketProductModel.getModesUrl().getVibrations() == null) {
            this.img_vibrate_mode.setVisibility(4);
            this.img_vibrate_mic.setVisibility(8);
            return;
        }
        if (socketProductModel.getModeStatus() == 1) {
            this.img_vibrate_mode.setVisibility(0);
        } else if (socketProductModel.getModeStatus() == 0) {
            this.img_vibrate_mode.setVisibility(4);
        }
        boolean equals2 = socketProductModel.getType().equals(Constans.UI_DEVICE_TYPE_CC_SMV);
        ImageView imageView2 = this.img_vibrate_mic;
        if (equals2) {
            imageView2.setVisibility(0);
        } else {
            imageView2.setVisibility(8);
        }
        this.mVibrateModes.clear();
        this.mVibrateModes.addAll(ArrayUtils.toArrayList(socketProductModel.getModesUrl().getVibrations()));
        this.mVibrateAdapter.notifyDataSetChanged();
        scrollVibrateHideImg();
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-fragment-PlaySMSFragment2, reason: not valid java name */
    /* synthetic */ void m601x366d75ee(int i, Permission permission) throws Exception {
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
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.9
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + PlaySMSFragment2.this.baseActivity.getPackageName()));
                    PlaySMSFragment2.this.startActivity(intent);
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
        if (view.getId() == R.id.layout_content) {
            if (this.layout_mode_list.getVisibility() == 0) {
                stopAnimation(this.layout_mode_list);
            }
            if (this.layout_vibrate_mode_list.getVisibility() == 0) {
                stopAnimation(this.layout_vibrate_mode_list);
                return;
            }
            return;
        }
        if (view.getId() == R.id.img_speed_mode) {
            if (this.inAnim) {
                return;
            }
            if (this.layout_mode_list.getVisibility() == 0) {
                stopAnimation(this.layout_mode_list);
                return;
            }
            this.isShowMode = 1;
            int visibility = this.layout_vibrate_mode_list.getVisibility();
            ConstraintLayout constraintLayout = this.layout_mode_list;
            if (visibility == 0) {
                startAnimation(constraintLayout, this.layout_vibrate_mode_list);
                return;
            } else {
                startAnimation(constraintLayout, null);
                return;
            }
        }
        if (view.getId() == R.id.img_vibrate_mode) {
            if (this.inAnim) {
                return;
            }
            if (this.layout_vibrate_mode_list.getVisibility() == 0) {
                stopAnimation(this.layout_vibrate_mode_list);
                return;
            }
            this.isShowMode = 2;
            int visibility2 = this.layout_mode_list.getVisibility();
            ConstraintLayout constraintLayout2 = this.layout_vibrate_mode_list;
            if (visibility2 == 0) {
                startAnimation(constraintLayout2, this.layout_mode_list);
                return;
            } else {
                startAnimation(constraintLayout2, null);
                return;
            }
        }
        if (view.getId() == R.id.img_close_mode) {
            SocketManager.getInstance().setMode(0);
            return;
        }
        if (view.getId() == R.id.img_close_vibrate_mode) {
            SocketManager.getInstance().setVibrateMode(0);
            return;
        }
        if (view.getId() == R.id.img_speed_mic) {
            initRecordPermission(1);
            return;
        }
        if (view.getId() == R.id.img_speed_shake) {
            boolean isShake = this.mBleSpeedUtil.isShake();
            BleSpeedUtil bleSpeedUtil = this.mBleSpeedUtil;
            if (isShake) {
                bleSpeedUtil.setShake(false);
                this.img_speed_shake.setImageResource(R.mipmap.ic_iplay_shake);
                return;
            } else {
                bleSpeedUtil.setShake(true);
                this.img_speed_shake.setImageResource(R.mipmap.ic_iplay_shake_select);
                return;
            }
        }
        if (view.getId() == R.id.img_vibrate_mic) {
            initRecordPermission(2);
            return;
        }
        if (view.getId() == R.id.img_vibrate_shake) {
            boolean isShake2 = this.mBleVibrateUtil.isShake();
            BleSpeedUtil bleSpeedUtil2 = this.mBleVibrateUtil;
            if (isShake2) {
                bleSpeedUtil2.setShake(false);
                this.img_vibrate_shake.setImageResource(R.mipmap.ic_iplay_shake);
                return;
            } else {
                bleSpeedUtil2.setShake(true);
                this.img_vibrate_shake.setImageResource(R.mipmap.ic_iplay_shake_select);
                return;
            }
        }
        if (view.getId() == R.id.layout_arrow_up) {
            if (this.recycler_mode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.recycler_mode.smoothScrollToPosition(0);
            return;
        }
        if (view.getId() == R.id.layout_arrow_down) {
            if (this.recycler_mode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.recycler_mode.smoothScrollToPosition(this.mAdapter.getItemCount() - 1);
            return;
        }
        if (view.getId() == R.id.layout_vibrate_arrow_up) {
            if (this.recycler_vibrate_mode == null || this.mVibrateAdapter.getItemCount() <= 1) {
                return;
            }
            this.recycler_vibrate_mode.smoothScrollToPosition(0);
            return;
        }
        if (view.getId() != R.id.layout_vibrate_arrow_down || this.recycler_vibrate_mode == null || this.mVibrateAdapter.getItemCount() <= 1) {
            return;
        }
        this.recycler_vibrate_mode.smoothScrollToPosition(this.mVibrateAdapter.getItemCount() - 1);
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sms, viewGroup, false);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        unregisterReceiver();
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.cancelRecording();
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onNotify(String str) {
        try {
            SocketPlayFragment socketPlayFragment = this.mFragment;
            if (socketPlayFragment != null && socketPlayFragment.isAdded()) {
                this.mFragment.onNotify(str);
            }
            SocketPlayFragment socketPlayFragment2 = this.YouFragment;
            if (socketPlayFragment2 != null && socketPlayFragment2.isAdded()) {
                this.YouFragment.onNotify(str);
            }
            initSocketData(str);
        } catch (Exception e) {
            e.printStackTrace();
        }
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
            SocketProductModel socketProductModel = this.mProductModel;
            int shake = (socketProductModel == null || socketProductModel.getShake() <= 0) ? 100 : this.mProductModel.getShake();
            this.mBleSpeedUtil.shakeSpeed(f, f2, f3, shake, new BleSpeedUtil.OnShakeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.14
                @Override // com.hismith.hismithapp.utils.BleSpeedUtil.OnShakeListener
                public void onShake(int i) {
                    SocketManager.getInstance().setSpeed(i);
                    PlaySMSFragment2.this.getBinderProduct().setSpeed(i);
                    PlaySMSFragment2.this.refreshSpeedModeUI(0);
                }
            });
            this.mBleVibrateUtil.shakeSpeed(f, f2, f3, shake, new BleSpeedUtil.OnShakeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment2.15
                @Override // com.hismith.hismithapp.utils.BleSpeedUtil.OnShakeListener
                public void onShake(int i) {
                    SocketManager.getInstance().setVibrate(i);
                    PlaySMSFragment2.this.getBinderProduct().setVspeed(i);
                    PlaySMSFragment2.this.refreshVibrateModeUI(0);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onSetSafeMax(int i) {
        SocketPlayFragment socketPlayFragment = this.mFragment;
        if (socketPlayFragment == null || !socketPlayFragment.isAdded()) {
            return;
        }
        this.mFragment.onSetSafeMax(i);
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onSetVibrateSafeMax(int i) {
        SocketPlayFragment socketPlayFragment = this.mFragment;
        if (socketPlayFragment == null || !socketPlayFragment.isAdded()) {
            return;
        }
        this.mFragment.onSetVibrateSafeMax(i);
    }

    public void refreshSpeedModeUI(int i) {
        Message message = new Message();
        message.what = 2;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
    }

    public void refreshVibrateModeUI(int i) {
        Message message = new Message();
        message.what = 4;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void updateSocketProductModel(SocketProductModel socketProductModel) {
        super.updateSocketProductModel(socketProductModel);
        this.mProductModel = socketProductModel;
        initFragment();
        initProduct(this.mProductModel);
    }
}
