package com.hismith.hismithapp.view;

import android.animation.ValueAnimator;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
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
import android.widget.SeekBar;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.facebook.appevents.integrity.IntegrityManager;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.databinding.LayoutDevicePanelBinding;
import com.hismith.hismithapp.fragment.LazyLoadFragment;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class PlayPanelView extends LazyLoadFragment {
    private float downX;
    private float endDownX;
    private View.OnTouchListener endOnTouch;
    private float funs2DownX;
    private View.OnTouchListener funs2SafeOnTouch;
    private LayoutDevicePanelBinding mBinding;
    private Context mContext;
    private int mFuns2SafeMax;
    private int mFuns2Value;
    private Handler mHandler;
    private int mPositionEnd;
    private int mPositionStart;
    private BroadcastReceiver mReceiver;
    private int mSafeMax;
    private int mSpeed;
    private onSafeChangeListener onSafeChangeListener;
    private View.OnTouchListener safeOnTouch;
    private float startDownX;
    private View.OnTouchListener startOnTouch;
    private int start_stop_position_diff;

    public interface onSafeChangeListener {
        void onFuns2Safe(int i);

        void onProductInfo();

        void onSpeedSafe(int i);
    }

    public PlayPanelView() {
        this.mSpeed = 0;
        this.mFuns2Value = 0;
        this.start_stop_position_diff = Constans.start_stop_position_diff;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.view.PlayPanelView.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                Bundle data = message.getData();
                String string = data != null ? data.getString(Constans.BleType) : "";
                int i = message.what;
                if (i == 3) {
                    PlayPanelView.this.refreshSpeedUI(((Integer) message.obj).intValue());
                    return false;
                }
                if (i == 5) {
                    PlayPanelView.this.refreshFuns2ValueUI(((Integer) message.obj).intValue());
                    return false;
                }
                switch (i) {
                    case 7:
                        int intValue = ((Integer) message.obj).intValue();
                        "CC".equals(string);
                        PlayPanelView.this.refreshStartBitUI(intValue);
                        break;
                    case 8:
                        int intValue2 = ((Integer) message.obj).intValue();
                        if (!"CC".equals(string)) {
                            PlayPanelView.this.refreshEndBitUI(intValue2);
                            break;
                        }
                        break;
                    case 9:
                        PlayPanelView.this.refreshSmoothUI(((Integer) message.obj).intValue());
                        break;
                    case 10:
                        PlayPanelView.this.refreshPositionUI(((Integer) message.obj).intValue());
                        break;
                }
                return false;
            }
        });
        this.mSafeMax = 100;
        this.safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.6
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlayPanelView.this.downX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayPanelView.this.downX;
                int right = PlayPanelView.this.mBinding.progressSpeed.getRight() + (PlayPanelView.this.mBinding.layoutSpeedMax.getWidth() / 2);
                LogUtil.e("getRight():" + PlayPanelView.this.mBinding.layoutSpeedMax.getRight() + ", width:" + PlayPanelView.this.mBinding.progressSpeed.getRight() + ", " + PlayPanelView.this.mBinding.layoutSpeedMax.getWidth());
                int right2 = (int) (right - (PlayPanelView.this.mBinding.layoutSpeedMax.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > PlayPanelView.this.mBinding.progressSpeed.getWidth()) {
                    right2 = PlayPanelView.this.mBinding.progressSpeed.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayPanelView.this.mBinding.layoutSpeedMax.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                PlayPanelView.this.mBinding.layoutSpeedMax.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / PlayPanelView.this.mBinding.progressSpeed.getWidth()));
                PlayPanelView.this.mSafeMax = width;
                PlayPanelView.this.mBinding.txtSpeedMax.setText(width + "");
                if (PlayPanelView.this.onSafeChangeListener != null) {
                    PlayPanelView.this.onSafeChangeListener.onSpeedSafe(width);
                }
                if (width < 0 || width >= PlayPanelView.this.mSpeed) {
                    return true;
                }
                BleManager.getInstance(PlayPanelView.this.mContext).writeSpeed(width);
                PlayPanelView.this.mSpeed = width;
                PlayPanelView.this.mBinding.progressSpeed.setProgress(width);
                PlayPanelView.this.mBinding.txtSpeed.setText(width + "");
                return true;
            }
        };
        this.mFuns2SafeMax = 100;
        this.funs2SafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.7
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlayPanelView.this.funs2DownX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayPanelView.this.funs2DownX;
                int right = PlayPanelView.this.mBinding.progressFuns2.getRight() + (PlayPanelView.this.mBinding.layoutFuns2Max.getWidth() / 2);
                LogUtil.e("getRight():" + PlayPanelView.this.mBinding.layoutFuns2Max.getRight() + ", width:" + PlayPanelView.this.mBinding.progressFuns2.getRight() + ", " + PlayPanelView.this.mBinding.layoutFuns2Max.getWidth());
                int right2 = (int) (right - (PlayPanelView.this.mBinding.layoutFuns2Max.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > PlayPanelView.this.mBinding.progressFuns2.getWidth()) {
                    right2 = PlayPanelView.this.mBinding.progressFuns2.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayPanelView.this.mBinding.layoutFuns2Max.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                PlayPanelView.this.mBinding.layoutFuns2Max.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / PlayPanelView.this.mBinding.progressFuns2.getWidth()));
                PlayPanelView.this.mFuns2SafeMax = width;
                PlayPanelView.this.mBinding.txtFuns2Max.setText(width + "");
                if (PlayPanelView.this.onSafeChangeListener != null) {
                    PlayPanelView.this.onSafeChangeListener.onFuns2Safe(width);
                }
                if (width < 0 || width >= PlayPanelView.this.mFuns2Value) {
                    return true;
                }
                BleManager.getInstance(PlayPanelView.this.mContext).writeVibrate(width);
                PlayPanelView.this.mFuns2Value = width;
                PlayPanelView.this.mBinding.progressFuns2.setProgress(width);
                PlayPanelView.this.mBinding.txtFuns2.setText(width + "");
                return true;
            }
        };
        this.mPositionEnd = 100;
        this.endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.8
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    view.bringToFront();
                    PlayPanelView.this.endDownX = motionEvent.getX();
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayPanelView.this.endDownX;
                int width = PlayPanelView.this.mBinding.progressPosition.getWidth();
                int right = (int) ((PlayPanelView.this.mBinding.progressPosition.getRight() + (PlayPanelView.this.mBinding.layoutPositionMax.getWidth() / 2)) - (PlayPanelView.this.mBinding.layoutPositionMax.getRight() + x));
                LogUtil.e("onTouch moveX:" + x + ", right():" + right);
                if (right < 0) {
                    right = 0;
                } else if (right > width) {
                    right = width;
                }
                double d = width;
                int i = (int) (100.0d - ((right * 100.0d) / d));
                if (i <= PlayPanelView.this.mPositionStart + PlayPanelView.this.start_stop_position_diff) {
                    int i2 = PlayPanelView.this.mPositionStart + PlayPanelView.this.start_stop_position_diff;
                    i = i2 > 100 ? 100 : i2;
                    right = (int) (d - ((d / 100.0d) * i));
                }
                int i3 = i <= 100 ? i < 0 ? 0 : i : 100;
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayPanelView.this.mBinding.layoutPositionMax.getLayoutParams();
                layoutParams.setMargins(0, 0, right, 0);
                PlayPanelView.this.mBinding.layoutPositionMax.setLayoutParams(layoutParams);
                if (PlayPanelView.this.mPositionEnd == i3) {
                    return true;
                }
                PlayPanelView.this.mPositionEnd = i3;
                LogUtil.e("结束位置：" + PlayPanelView.this.mPositionEnd + ", right:" + right);
                PlayPanelView.this.mBinding.txtPositionMax.setText(i3 + "");
                BleManager.getInstance(PlayPanelView.this.mContext).writeStopPosition(PlayPanelView.this.mPositionEnd);
                return true;
            }
        };
        this.mPositionStart = 0;
        this.startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.9
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action != 0) {
                    if (action != 2) {
                        return true;
                    }
                    PlayPanelView.this.touchStartPosition(motionEvent.getX());
                    return true;
                }
                view.bringToFront();
                PlayPanelView.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
        };
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.view.PlayPanelView.10
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();
                if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                    PlayPanelView.this.initProduct();
                    if (PlayPanelView.this.onSafeChangeListener != null) {
                        PlayPanelView.this.onSafeChangeListener.onProductInfo();
                        return;
                    }
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
                        PlayPanelView.this.initBleAAData(stringExtra2, stringExtra3);
                    }
                    if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                        PlayPanelView.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                    }
                }
            }
        };
    }

    public PlayPanelView(int i, int i2) {
        this.mSpeed = 0;
        this.mFuns2Value = 0;
        this.start_stop_position_diff = Constans.start_stop_position_diff;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.view.PlayPanelView.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                Bundle data = message.getData();
                String string = data != null ? data.getString(Constans.BleType) : "";
                int i3 = message.what;
                if (i3 == 3) {
                    PlayPanelView.this.refreshSpeedUI(((Integer) message.obj).intValue());
                    return false;
                }
                if (i3 == 5) {
                    PlayPanelView.this.refreshFuns2ValueUI(((Integer) message.obj).intValue());
                    return false;
                }
                switch (i3) {
                    case 7:
                        int intValue = ((Integer) message.obj).intValue();
                        "CC".equals(string);
                        PlayPanelView.this.refreshStartBitUI(intValue);
                        break;
                    case 8:
                        int intValue2 = ((Integer) message.obj).intValue();
                        if (!"CC".equals(string)) {
                            PlayPanelView.this.refreshEndBitUI(intValue2);
                            break;
                        }
                        break;
                    case 9:
                        PlayPanelView.this.refreshSmoothUI(((Integer) message.obj).intValue());
                        break;
                    case 10:
                        PlayPanelView.this.refreshPositionUI(((Integer) message.obj).intValue());
                        break;
                }
                return false;
            }
        });
        this.mSafeMax = 100;
        this.safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.6
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlayPanelView.this.downX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayPanelView.this.downX;
                int right = PlayPanelView.this.mBinding.progressSpeed.getRight() + (PlayPanelView.this.mBinding.layoutSpeedMax.getWidth() / 2);
                LogUtil.e("getRight():" + PlayPanelView.this.mBinding.layoutSpeedMax.getRight() + ", width:" + PlayPanelView.this.mBinding.progressSpeed.getRight() + ", " + PlayPanelView.this.mBinding.layoutSpeedMax.getWidth());
                int right2 = (int) (right - (PlayPanelView.this.mBinding.layoutSpeedMax.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > PlayPanelView.this.mBinding.progressSpeed.getWidth()) {
                    right2 = PlayPanelView.this.mBinding.progressSpeed.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayPanelView.this.mBinding.layoutSpeedMax.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                PlayPanelView.this.mBinding.layoutSpeedMax.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / PlayPanelView.this.mBinding.progressSpeed.getWidth()));
                PlayPanelView.this.mSafeMax = width;
                PlayPanelView.this.mBinding.txtSpeedMax.setText(width + "");
                if (PlayPanelView.this.onSafeChangeListener != null) {
                    PlayPanelView.this.onSafeChangeListener.onSpeedSafe(width);
                }
                if (width < 0 || width >= PlayPanelView.this.mSpeed) {
                    return true;
                }
                BleManager.getInstance(PlayPanelView.this.mContext).writeSpeed(width);
                PlayPanelView.this.mSpeed = width;
                PlayPanelView.this.mBinding.progressSpeed.setProgress(width);
                PlayPanelView.this.mBinding.txtSpeed.setText(width + "");
                return true;
            }
        };
        this.mFuns2SafeMax = 100;
        this.funs2SafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.7
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlayPanelView.this.funs2DownX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayPanelView.this.funs2DownX;
                int right = PlayPanelView.this.mBinding.progressFuns2.getRight() + (PlayPanelView.this.mBinding.layoutFuns2Max.getWidth() / 2);
                LogUtil.e("getRight():" + PlayPanelView.this.mBinding.layoutFuns2Max.getRight() + ", width:" + PlayPanelView.this.mBinding.progressFuns2.getRight() + ", " + PlayPanelView.this.mBinding.layoutFuns2Max.getWidth());
                int right2 = (int) (right - (PlayPanelView.this.mBinding.layoutFuns2Max.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > PlayPanelView.this.mBinding.progressFuns2.getWidth()) {
                    right2 = PlayPanelView.this.mBinding.progressFuns2.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayPanelView.this.mBinding.layoutFuns2Max.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                PlayPanelView.this.mBinding.layoutFuns2Max.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / PlayPanelView.this.mBinding.progressFuns2.getWidth()));
                PlayPanelView.this.mFuns2SafeMax = width;
                PlayPanelView.this.mBinding.txtFuns2Max.setText(width + "");
                if (PlayPanelView.this.onSafeChangeListener != null) {
                    PlayPanelView.this.onSafeChangeListener.onFuns2Safe(width);
                }
                if (width < 0 || width >= PlayPanelView.this.mFuns2Value) {
                    return true;
                }
                BleManager.getInstance(PlayPanelView.this.mContext).writeVibrate(width);
                PlayPanelView.this.mFuns2Value = width;
                PlayPanelView.this.mBinding.progressFuns2.setProgress(width);
                PlayPanelView.this.mBinding.txtFuns2.setText(width + "");
                return true;
            }
        };
        this.mPositionEnd = 100;
        this.endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.8
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    view.bringToFront();
                    PlayPanelView.this.endDownX = motionEvent.getX();
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayPanelView.this.endDownX;
                int width = PlayPanelView.this.mBinding.progressPosition.getWidth();
                int right = (int) ((PlayPanelView.this.mBinding.progressPosition.getRight() + (PlayPanelView.this.mBinding.layoutPositionMax.getWidth() / 2)) - (PlayPanelView.this.mBinding.layoutPositionMax.getRight() + x));
                LogUtil.e("onTouch moveX:" + x + ", right():" + right);
                if (right < 0) {
                    right = 0;
                } else if (right > width) {
                    right = width;
                }
                double d = width;
                int i3 = (int) (100.0d - ((right * 100.0d) / d));
                if (i3 <= PlayPanelView.this.mPositionStart + PlayPanelView.this.start_stop_position_diff) {
                    int i22 = PlayPanelView.this.mPositionStart + PlayPanelView.this.start_stop_position_diff;
                    i3 = i22 > 100 ? 100 : i22;
                    right = (int) (d - ((d / 100.0d) * i3));
                }
                int i32 = i3 <= 100 ? i3 < 0 ? 0 : i3 : 100;
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayPanelView.this.mBinding.layoutPositionMax.getLayoutParams();
                layoutParams.setMargins(0, 0, right, 0);
                PlayPanelView.this.mBinding.layoutPositionMax.setLayoutParams(layoutParams);
                if (PlayPanelView.this.mPositionEnd == i32) {
                    return true;
                }
                PlayPanelView.this.mPositionEnd = i32;
                LogUtil.e("结束位置：" + PlayPanelView.this.mPositionEnd + ", right:" + right);
                PlayPanelView.this.mBinding.txtPositionMax.setText(i32 + "");
                BleManager.getInstance(PlayPanelView.this.mContext).writeStopPosition(PlayPanelView.this.mPositionEnd);
                return true;
            }
        };
        this.mPositionStart = 0;
        this.startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.9
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action != 0) {
                    if (action != 2) {
                        return true;
                    }
                    PlayPanelView.this.touchStartPosition(motionEvent.getX());
                    return true;
                }
                view.bringToFront();
                PlayPanelView.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
        };
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.view.PlayPanelView.10
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();
                if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                    PlayPanelView.this.initProduct();
                    if (PlayPanelView.this.onSafeChangeListener != null) {
                        PlayPanelView.this.onSafeChangeListener.onProductInfo();
                        return;
                    }
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
                        PlayPanelView.this.initBleAAData(stringExtra2, stringExtra3);
                    }
                    if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                        PlayPanelView.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                    }
                }
            }
        };
        this.mSafeMax = i;
        this.mFuns2SafeMax = i2;
    }

    private void init() {
        BleManager.getInstance(this.mContext).initWriteMessage();
        initProduct();
        initBleDataUI();
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleAAData(String str, String str2) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        str.hashCode();
        if (str.equals("03") || str.equals("04")) {
            try {
                int intValue = Integer.valueOf(str2, 16).intValue();
                if (intValue > 100) {
                    intValue = 100;
                } else if (intValue < 0) {
                    intValue = 0;
                }
                Message message = new Message();
                message.what = 3;
                message.obj = Integer.valueOf(intValue);
                this.mHandler.sendMessage(message);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void initBleDataUI() {
        if (BleManager.getInstance(this.mContext).getmConnBleDevice() != null) {
            Message message = new Message();
            message.what = 3;
            message.obj = Integer.valueOf(BleManager.getInstance(this.mContext).getmConnBleDevice().getSpeed());
            this.mHandler.sendMessage(message);
            int startPosition = BleManager.getInstance(this.mContext).getmConnBleDevice().getStartPosition();
            Message message2 = new Message();
            message2.what = 7;
            message2.obj = Integer.valueOf(startPosition);
            this.mHandler.sendMessage(message2);
            int stopPosition = BleManager.getInstance(this.mContext).getmConnBleDevice().getStopPosition();
            Message message3 = new Message();
            message3.what = 8;
            if (stopPosition <= 0) {
                stopPosition = this.start_stop_position_diff + startPosition;
            }
            message3.obj = Integer.valueOf(stopPosition);
            this.mHandler.sendMessage(message3);
            Message message4 = new Message();
            message4.what = 9;
            message4.obj = Integer.valueOf(BleManager.getInstance(this.mContext).getmConnBleDevice().getSmoothness());
            this.mHandler.sendMessage(message4);
        }
        this.mBinding.layoutSpeedMax.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.3
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlayPanelView.this.mBinding.layoutSpeedMax.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                PlayPanelView playPanelView = PlayPanelView.this;
                playPanelView.refreshSafeUI(playPanelView.mSafeMax);
            }
        });
        this.mBinding.layoutFuns2Max.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.4
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlayPanelView.this.mBinding.layoutFuns2Max.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                PlayPanelView playPanelView = PlayPanelView.this;
                playPanelView.refreshFuns2SafeUI(playPanelView.mFuns2SafeMax);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initProduct() {
        if (ProductUtil.isSMSType()) {
            this.mBinding.groupSmv.setVisibility(8);
            this.mBinding.groupSms.setVisibility(0);
            return;
        }
        boolean isSMVType = ProductUtil.isSMVType();
        LayoutDevicePanelBinding layoutDevicePanelBinding = this.mBinding;
        if (isSMVType) {
            layoutDevicePanelBinding.groupSmv.setVisibility(0);
            this.mBinding.groupSms.setVisibility(8);
        } else {
            layoutDevicePanelBinding.groupSmv.setVisibility(8);
            this.mBinding.groupSms.setVisibility(8);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        if (this.mPositionEnd != i) {
            this.mPositionEnd = i;
            this.mBinding.txtPositionMax.setText(this.mPositionEnd + "");
            double width = this.mBinding.progressPosition.getWidth();
            int i2 = (int) (width - ((width / 100.0d) * i));
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.mBinding.layoutPositionMax.getLayoutParams();
            layoutParams.setMargins(0, 0, i2, 0);
            this.mBinding.layoutPositionMax.setLayoutParams(layoutParams);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshFuns2SafeUI(int i) {
        this.mFuns2SafeMax = i;
        this.mBinding.txtFuns2Max.setText(this.mFuns2SafeMax + "");
        double width = this.mBinding.progressFuns2.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * this.mFuns2SafeMax));
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.mBinding.layoutFuns2Max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.mBinding.layoutFuns2Max.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshFuns2ValueUI(int i) {
        this.mFuns2Value = i;
        this.mBinding.progressFuns2.setProgress(i);
        this.mBinding.txtFuns2.setText(i + "");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionUI(int i) {
        this.mBinding.txtPosition.setText(i + "");
        this.mBinding.progressPosition.setProgress(i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.mSafeMax = i;
        this.mBinding.txtSpeedMax.setText(this.mSafeMax + "");
        double width = this.mBinding.progressSpeed.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * this.mSafeMax));
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.mBinding.layoutSpeedMax.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.mBinding.layoutSpeedMax.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSmoothUI(int i) {
        this.mBinding.txtSmooth.setText(i + "");
        ValueAnimator ofInt = ValueAnimator.ofInt(this.mBinding.seekSmooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.5
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                PlayPanelView.this.mBinding.seekSmooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
            }
        });
        ofInt.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedUI(int i) {
        this.mSpeed = i;
        this.mBinding.progressSpeed.setProgress(i);
        this.mBinding.txtSpeed.setText(i + "");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartBitUI(int i) {
        if (this.mPositionStart != i) {
            this.mPositionStart = i;
            this.mBinding.txtPositionMin.setText(this.mPositionStart + "");
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.mBinding.layoutPositionMin.getLayoutParams();
            layoutParams.setMargins((int) ((this.mBinding.progressPosition.getWidth() / 100.0d) * i), 0, 0, 0);
            this.mBinding.layoutPositionMin.setLayoutParams(layoutParams);
        }
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.mContext, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.mBinding.layoutSpeedMax.setOnTouchListener(this.safeOnTouch);
        this.mBinding.layoutFuns2Max.setOnTouchListener(this.funs2SafeOnTouch);
        this.mBinding.layoutPositionMax.setOnTouchListener(this.endOnTouch);
        this.mBinding.layoutPositionMin.setOnTouchListener(this.startOnTouch);
        this.mBinding.seekSmooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.2
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + PlayPanelView.this.mBinding.seekSmooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + PlayPanelView.this.mBinding.seekSmooth.getProgress());
                int progress = PlayPanelView.this.mBinding.seekSmooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                int i3 = i / 100;
                PlayPanelView.this.mBinding.txtSmooth.setText(i3 + "");
                BleManager.getInstance(PlayPanelView.this.mContext).writeSmooth(i3);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.PlayPanelView.2.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        PlayPanelView.this.mBinding.seekSmooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void touchStartPosition(float f) {
        float f2 = f - this.startDownX;
        int width = this.mBinding.progressPosition.getWidth();
        int left = (int) ((this.mBinding.layoutPositionMin.getLeft() + f2) - (this.mBinding.progressPosition.getLeft() - (this.mBinding.layoutPositionMin.getWidth() / 2)));
        if (left < 0) {
            left = 0;
        } else if (left > width) {
            left = width;
        }
        double d = width;
        int i = (int) ((left * 100.0d) / d);
        int i2 = this.mPositionEnd;
        int i3 = this.start_stop_position_diff;
        if (i >= i2 - i3) {
            int i4 = i2 - i3;
            i = i4 < 0 ? 0 : i4;
            left = (int) ((d / 100.0d) * i);
        }
        if (i > 100) {
            i = 100;
        } else if (i < 0) {
            i = 0;
        }
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.mBinding.layoutPositionMin.getLayoutParams();
        layoutParams.setMargins(left, 0, 0, 0);
        this.mBinding.layoutPositionMin.setLayoutParams(layoutParams);
        if (this.mPositionStart != i) {
            this.mPositionStart = i;
            LogUtil.e("开始位置：" + this.mPositionStart + ", left:" + left);
            this.mBinding.txtPositionMin.setText(i + "");
            BleManager.getInstance(this.mContext).writeStartPosition(this.mPositionStart);
        }
    }

    private void unregisterReceiver() {
        try {
            this.mContext.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void initBleCCData(String str, String str2, String str3) {
        try {
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

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mContext = getContext();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        LayoutDevicePanelBinding inflate = LayoutDevicePanelBinding.inflate(layoutInflater, viewGroup, false);
        this.mBinding = inflate;
        LanguageUtil.setText(this.mContext, inflate.txtSpeedLabel, "speedTitle");
        this.mBinding.txtPositionLabel.setText(LanguageUtil.getLanguageByKey(this.mContext, "position") + ":");
        this.mBinding.txtSmoothLabel.setText(LanguageUtil.getLanguageByKey(this.mContext, "smoothness") + ":");
        return this.mBinding.getRoot();
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        this.mHandler.removeCallbacksAndMessages(null);
        unregisterReceiver();
    }

    public void setOnSafeChangeListener(onSafeChangeListener onsafechangelistener) {
        this.onSafeChangeListener = onsafechangelistener;
    }

    public void setmFuns2SafeMax(int i) {
        this.mFuns2SafeMax = i;
    }

    public void setmSafeMax(int i) {
        this.mSafeMax = i;
    }
}
