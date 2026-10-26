package com.hismith.hismithapp.view.plug;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.util.DisplayMetrics;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Group;
import com.facebook.appevents.integrity.IntegrityManager;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.PiuTimerDialog;
import com.hismith.hismithapp.fragment.LazyLoadFragment;
import com.hismith.hismithapp.manager.PiuTimerManager;
import com.hismith.hismithapp.model.PiuTimerModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.GradientRoundProgress;

/* loaded from: classes3.dex */
public class PlugPiuTimer extends LazyLoadFragment implements View.OnClickListener {
    private final int WHAT_BATTERY;
    private final int WHAT_PIU;
    private final int WHAT_PIU_TEXT;
    private final int WHAT_WRITE_PIU;
    private final int WHAT_WRITE_PIU_ONE;
    private View.OnTouchListener dragOnTouchListener;
    private int isPiuTimerPlay;
    private int lastDown;
    private Activity mActivity;
    private Context mContext;
    private float mDialogHeightPercent;
    private Group mGroupPiu;
    private Handler mHandler;
    private ImageView mImgBattery;
    private ImageView mImgPiu;
    private ImageView mImgPiuTime;
    private ConstraintLayout mLayoutBattery;
    private FrameLayout mLayoutPiuBtn;
    private PiuTimerDialog mPiuTimerDialog;
    private PiuTimerModel mPiuTimerModel;
    private GradientRoundProgress mProgressPiu;
    private BroadcastReceiver mReceiver;
    private TextView mTxtPiuTime;
    private View mViewBattery;
    private View.OnTouchListener piuOnTouchListener;
    private View rootView;

    public PlugPiuTimer() {
        this.WHAT_PIU = 11;
        this.WHAT_BATTERY = 13;
        this.WHAT_WRITE_PIU = 12;
        this.WHAT_WRITE_PIU_ONE = 18;
        this.WHAT_PIU_TEXT = 17;
        this.isPiuTimerPlay = 0;
        this.mDialogHeightPercent = 0.86f;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int i = message.what;
                if (i == 17) {
                    LanguageUtil.setText(PlugPiuTimer.this.mContext, PlugPiuTimer.this.mTxtPiuTime, "spray");
                    return false;
                }
                if (i == 18) {
                    LogUtil.e("piuTimer喷射!");
                    BleManager.getInstance(PlugPiuTimer.this.mContext).writeOpenPiu();
                    return false;
                }
                switch (i) {
                    case 11:
                        int intValue = ((Integer) message.obj).intValue();
                        PlugPiuTimer plugPiuTimer = PlugPiuTimer.this;
                        if (intValue != 1) {
                            plugPiuTimer.mImgPiu.setImageResource(R.mipmap.ic_iplay_piu);
                            break;
                        } else {
                            plugPiuTimer.mImgPiu.setImageResource(R.mipmap.ic_iplay_piu_select);
                            break;
                        }
                    case 12:
                        LogUtil.e("piuTimer喷射!");
                        BleManager.getInstance(PlugPiuTimer.this.mContext).writeOpenPiu();
                        PlugPiuTimer.this.mHandler.sendEmptyMessageDelayed(12, 100L);
                        break;
                    case 13:
                        PlugPiuTimer.this.changeBatteryUI();
                        break;
                }
                return false;
            }
        });
        this.dragOnTouchListener = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.3
            private static final int DRAG_THRESHOLD = 10;
            private boolean isDragging;
            private float startTranslationY;
            private float startY;

            /* JADX WARN: Code restructure failed: missing block: B:7:0x0024, code lost:
            
                if (r0 != 3) goto L23;
             */
            @Override // android.view.View.OnTouchListener
            /*
                Code decompiled incorrectly, please refer to instructions dump.
            */
            public boolean onTouch(View view, MotionEvent motionEvent) {
                LogUtil.e("drag拖动 action:" + motionEvent.getAction());
                int action = motionEvent.getAction();
                if (action != 0) {
                    if (action != 1) {
                        if (action == 2) {
                            float rawY = motionEvent.getRawY() - this.startY;
                            if (Math.abs(rawY) > 10.0f) {
                                this.isDragging = true;
                                float f = this.startTranslationY + rawY;
                                float maxTranslationY = PlugPiuTimer.this.getMaxTranslationY(view);
                                if (f < 0.0f) {
                                    f = 0.0f;
                                } else if (f > maxTranslationY) {
                                    f = maxTranslationY;
                                }
                                view.setTranslationY(f);
                            }
                        }
                    }
                    view.getParent().requestDisallowInterceptTouchEvent(false);
                    if (!this.isDragging) {
                        return false;
                    }
                } else {
                    this.startY = motionEvent.getRawY();
                    this.startTranslationY = view.getTranslationY();
                    this.isDragging = false;
                }
                return true;
            }
        };
        this.piuOnTouchListener = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlugPiuTimer.this.mHandler.sendEmptyMessage(12);
                } else if (action == 1) {
                    PlugPiuTimer.this.mHandler.removeMessages(12);
                }
                return true;
            }
        };
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.6
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                PiuTimerModel piuTimerModel;
                String action = intent.getAction();
                if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                    LogUtil.e("广播 RECEIVER_PRODUCT_INFO");
                    PlugPiuTimer.this.initProductData();
                    return;
                }
                if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                    String stringExtra = intent.getStringExtra(Constans.BleType);
                    String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                    String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                    String stringExtra4 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                    LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                    if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                        PlugPiuTimer.this.initBleCCData(stringExtra4, stringExtra2, stringExtra3);
                        return;
                    }
                    return;
                }
                if (Constans.RECEIVER_PIU_TIMER_ACTION.equals(action)) {
                    try {
                        String stringExtra5 = intent.getStringExtra(Constans.EXTRA_PIU_TIMER_ACTION);
                        if (!stringExtra5.equals(Constans.TIMER_ACTION_PLAY)) {
                            if (stringExtra5.equals(Constans.TIMER_ACTION_PAUSE)) {
                                LogUtil.e("piuTimer广播：" + stringExtra5);
                                PlugPiuTimer.this.isPiuTimerPlay = 2;
                                PlugPiuTimer.this.pausePiuTask();
                                if (PlugPiuTimer.this.mPiuTimerDialog != null) {
                                    PlugPiuTimer.this.mPiuTimerDialog.setPlay(false);
                                    return;
                                }
                                return;
                            }
                            if (stringExtra5.equals(Constans.TIMER_ACTION_STOP)) {
                                PlugPiuTimer.this.isPiuTimerPlay = 0;
                                PlugPiuTimer.this.stopPiuTask();
                                if (PlugPiuTimer.this.mPiuTimerDialog != null) {
                                    PlugPiuTimer.this.mPiuTimerDialog.setPlay(false);
                                    return;
                                }
                                return;
                            }
                            return;
                        }
                        if ((intent.getSerializableExtra(Constans.EXTRA_PIU_TIMER_MODEL) instanceof PiuTimerModel) && (piuTimerModel = (PiuTimerModel) intent.getSerializableExtra(Constans.EXTRA_PIU_TIMER_MODEL)) != null) {
                            LogUtil.e("piuTimer广播：" + stringExtra5 + ", 时间：" + piuTimerModel.getSecond());
                            if (PlugPiuTimer.this.mPiuTimerModel == null || PlugPiuTimer.this.mPiuTimerModel.getUpdateTimeMillis() != piuTimerModel.getUpdateTimeMillis() || piuTimerModel.getSecond() == 0 || PlugPiuTimer.this.isPiuTimerPlay != 2) {
                                PlugPiuTimer.this.isPiuTimerPlay = 1;
                                PlugPiuTimer.this.mPiuTimerModel = piuTimerModel;
                                PlugPiuTimer.this.stopPiuTask();
                                PlugPiuTimer.this.startPiuTask();
                            } else {
                                PlugPiuTimer.this.isPiuTimerPlay = 1;
                                PlugPiuTimer.this.resumePiuTask();
                            }
                            if (PlugPiuTimer.this.mPiuTimerDialog != null) {
                                PlugPiuTimer.this.mPiuTimerDialog.setPlay(PlugPiuTimer.this.isPiuTimerPlay == 1);
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        this.lastDown = 0;
    }

    public PlugPiuTimer(float f) {
        this.WHAT_PIU = 11;
        this.WHAT_BATTERY = 13;
        this.WHAT_WRITE_PIU = 12;
        this.WHAT_WRITE_PIU_ONE = 18;
        this.WHAT_PIU_TEXT = 17;
        this.isPiuTimerPlay = 0;
        this.mDialogHeightPercent = 0.86f;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int i = message.what;
                if (i == 17) {
                    LanguageUtil.setText(PlugPiuTimer.this.mContext, PlugPiuTimer.this.mTxtPiuTime, "spray");
                    return false;
                }
                if (i == 18) {
                    LogUtil.e("piuTimer喷射!");
                    BleManager.getInstance(PlugPiuTimer.this.mContext).writeOpenPiu();
                    return false;
                }
                switch (i) {
                    case 11:
                        int intValue = ((Integer) message.obj).intValue();
                        PlugPiuTimer plugPiuTimer = PlugPiuTimer.this;
                        if (intValue != 1) {
                            plugPiuTimer.mImgPiu.setImageResource(R.mipmap.ic_iplay_piu);
                            break;
                        } else {
                            plugPiuTimer.mImgPiu.setImageResource(R.mipmap.ic_iplay_piu_select);
                            break;
                        }
                    case 12:
                        LogUtil.e("piuTimer喷射!");
                        BleManager.getInstance(PlugPiuTimer.this.mContext).writeOpenPiu();
                        PlugPiuTimer.this.mHandler.sendEmptyMessageDelayed(12, 100L);
                        break;
                    case 13:
                        PlugPiuTimer.this.changeBatteryUI();
                        break;
                }
                return false;
            }
        });
        this.dragOnTouchListener = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.3
            private static final int DRAG_THRESHOLD = 10;
            private boolean isDragging;
            private float startTranslationY;
            private float startY;

            /* JADX WARN: Code restructure failed: missing block: B:7:0x0024, code lost:
            
                if (r0 != 3) goto L23;
             */
            @Override // android.view.View.OnTouchListener
            /*
                Code decompiled incorrectly, please refer to instructions dump.
            */
            public boolean onTouch(View view, MotionEvent motionEvent) {
                LogUtil.e("drag拖动 action:" + motionEvent.getAction());
                int action = motionEvent.getAction();
                if (action != 0) {
                    if (action != 1) {
                        if (action == 2) {
                            float rawY = motionEvent.getRawY() - this.startY;
                            if (Math.abs(rawY) > 10.0f) {
                                this.isDragging = true;
                                float f2 = this.startTranslationY + rawY;
                                float maxTranslationY = PlugPiuTimer.this.getMaxTranslationY(view);
                                if (f2 < 0.0f) {
                                    f2 = 0.0f;
                                } else if (f2 > maxTranslationY) {
                                    f2 = maxTranslationY;
                                }
                                view.setTranslationY(f2);
                            }
                        }
                    }
                    view.getParent().requestDisallowInterceptTouchEvent(false);
                    if (!this.isDragging) {
                        return false;
                    }
                } else {
                    this.startY = motionEvent.getRawY();
                    this.startTranslationY = view.getTranslationY();
                    this.isDragging = false;
                }
                return true;
            }
        };
        this.piuOnTouchListener = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    PlugPiuTimer.this.mHandler.sendEmptyMessage(12);
                } else if (action == 1) {
                    PlugPiuTimer.this.mHandler.removeMessages(12);
                }
                return true;
            }
        };
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.6
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                PiuTimerModel piuTimerModel;
                String action = intent.getAction();
                if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                    LogUtil.e("广播 RECEIVER_PRODUCT_INFO");
                    PlugPiuTimer.this.initProductData();
                    return;
                }
                if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                    String stringExtra = intent.getStringExtra(Constans.BleType);
                    String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                    String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                    String stringExtra4 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                    LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                    if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                        PlugPiuTimer.this.initBleCCData(stringExtra4, stringExtra2, stringExtra3);
                        return;
                    }
                    return;
                }
                if (Constans.RECEIVER_PIU_TIMER_ACTION.equals(action)) {
                    try {
                        String stringExtra5 = intent.getStringExtra(Constans.EXTRA_PIU_TIMER_ACTION);
                        if (!stringExtra5.equals(Constans.TIMER_ACTION_PLAY)) {
                            if (stringExtra5.equals(Constans.TIMER_ACTION_PAUSE)) {
                                LogUtil.e("piuTimer广播：" + stringExtra5);
                                PlugPiuTimer.this.isPiuTimerPlay = 2;
                                PlugPiuTimer.this.pausePiuTask();
                                if (PlugPiuTimer.this.mPiuTimerDialog != null) {
                                    PlugPiuTimer.this.mPiuTimerDialog.setPlay(false);
                                    return;
                                }
                                return;
                            }
                            if (stringExtra5.equals(Constans.TIMER_ACTION_STOP)) {
                                PlugPiuTimer.this.isPiuTimerPlay = 0;
                                PlugPiuTimer.this.stopPiuTask();
                                if (PlugPiuTimer.this.mPiuTimerDialog != null) {
                                    PlugPiuTimer.this.mPiuTimerDialog.setPlay(false);
                                    return;
                                }
                                return;
                            }
                            return;
                        }
                        if ((intent.getSerializableExtra(Constans.EXTRA_PIU_TIMER_MODEL) instanceof PiuTimerModel) && (piuTimerModel = (PiuTimerModel) intent.getSerializableExtra(Constans.EXTRA_PIU_TIMER_MODEL)) != null) {
                            LogUtil.e("piuTimer广播：" + stringExtra5 + ", 时间：" + piuTimerModel.getSecond());
                            if (PlugPiuTimer.this.mPiuTimerModel == null || PlugPiuTimer.this.mPiuTimerModel.getUpdateTimeMillis() != piuTimerModel.getUpdateTimeMillis() || piuTimerModel.getSecond() == 0 || PlugPiuTimer.this.isPiuTimerPlay != 2) {
                                PlugPiuTimer.this.isPiuTimerPlay = 1;
                                PlugPiuTimer.this.mPiuTimerModel = piuTimerModel;
                                PlugPiuTimer.this.stopPiuTask();
                                PlugPiuTimer.this.startPiuTask();
                            } else {
                                PlugPiuTimer.this.isPiuTimerPlay = 1;
                                PlugPiuTimer.this.resumePiuTask();
                            }
                            if (PlugPiuTimer.this.mPiuTimerDialog != null) {
                                PlugPiuTimer.this.mPiuTimerDialog.setPlay(PlugPiuTimer.this.isPiuTimerPlay == 1);
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        this.lastDown = 0;
        this.mDialogHeightPercent = f;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void changeBatteryLayoutParams() {
        ViewGroup.LayoutParams layoutParams = this.mImgBattery.getLayoutParams();
        int width = (int) (this.mViewBattery.getWidth() * (BleManager.getInstance(this.mContext).getmConnBleDevice().getBattery() / 10.0d));
        ImageView imageView = this.mImgBattery;
        if (width >= 0) {
            imageView.setVisibility(0);
        } else {
            imageView.setVisibility(8);
        }
        layoutParams.width = width;
        this.mImgBattery.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void changeBatteryUI() {
        try {
            if (BleManager.getInstance(this.mContext).getmConnBleDevice() == null || BleManager.getInstance(this.mContext).getmConnBleDevice().getBattery() <= -1) {
                return;
            }
            if (this.mLayoutBattery.getVisibility() == 0) {
                changeBatteryLayoutParams();
            } else {
                this.mLayoutBattery.setVisibility(0);
                this.mViewBattery.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.5
                    @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
                    public void onGlobalLayout() {
                        PlugPiuTimer.this.mViewBattery.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                        PlugPiuTimer.this.changeBatteryLayoutParams();
                    }
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public float getMaxTranslationY(View view) {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        this.mActivity.getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int i = displayMetrics.heightPixels;
        int height = view.getHeight();
        if (height == 0) {
            view.measure(0, 0);
            height = view.getMeasuredHeight();
        }
        return Math.max(0, i - height);
    }

    private void init() {
        try {
            this.mLayoutBattery.setVisibility(8);
            boolean hasPiuFunBle = BleManager.getInstance(this.mContext).hasPiuFunBle();
            Group group = this.mGroupPiu;
            if (hasPiuFunBle) {
                group.setVisibility(0);
                if (BleManager.getInstance(this.mContext).getmConnBleDevice() != null && BleManager.getInstance(this.mContext).getmConnBleDevice().getBattery() > -1) {
                    this.mLayoutBattery.setVisibility(0);
                    this.mViewBattery.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.2
                        @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
                        public void onGlobalLayout() {
                            PlugPiuTimer.this.mViewBattery.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                            ViewGroup.LayoutParams layoutParams = PlugPiuTimer.this.mImgBattery.getLayoutParams();
                            if (BleManager.getInstance(PlugPiuTimer.this.mContext).getmConnBleDevice() != null) {
                                int width = (int) (PlugPiuTimer.this.mViewBattery.getWidth() * (BleManager.getInstance(PlugPiuTimer.this.mContext).getmConnBleDevice().getBattery() / 10.0d));
                                PlugPiuTimer plugPiuTimer = PlugPiuTimer.this;
                                if (width >= 0) {
                                    plugPiuTimer.mImgBattery.setVisibility(0);
                                } else {
                                    plugPiuTimer.mImgBattery.setVisibility(8);
                                }
                                layoutParams.width = width;
                                PlugPiuTimer.this.mImgBattery.setLayoutParams(layoutParams);
                            }
                        }
                    });
                }
            } else {
                group.setVisibility(8);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2, String str3) {
        if (TextUtils.isEmpty(str2) || TextUtils.isEmpty(str3)) {
            return;
        }
        str2.hashCode();
        if (str2.equals("00")) {
            if (MyApplication.getApplication().getDeviceAddress().equals(str)) {
                Message message = new Message();
                message.what = 13;
                this.mHandler.sendMessage(message);
                return;
            }
            return;
        }
        if (str2.equals(Constans.piuCommand)) {
            int intValue = Integer.valueOf(str3, 16).intValue();
            Message message2 = new Message();
            message2.what = 11;
            message2.obj = Integer.valueOf(intValue);
            this.mHandler.sendMessage(message2);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initProductData() {
        this.mLayoutBattery.setVisibility(8);
        boolean hasPiuFunBle = BleManager.getInstance(this.mContext).hasPiuFunBle();
        Group group = this.mGroupPiu;
        if (!hasPiuFunBle) {
            group.setVisibility(8);
            return;
        }
        group.setVisibility(0);
        if (BleManager.getInstance(this.mContext).getmConnBleDevice() == null || BleManager.getInstance(this.mContext).getmConnBleDevice().getBattery() <= -1) {
            return;
        }
        this.mLayoutBattery.setVisibility(0);
        this.mViewBattery.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.7
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlugPiuTimer.this.mViewBattery.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                ViewGroup.LayoutParams layoutParams = PlugPiuTimer.this.mImgBattery.getLayoutParams();
                if (BleManager.getInstance(PlugPiuTimer.this.mContext).getmConnBleDevice() != null) {
                    int width = (int) (PlugPiuTimer.this.mViewBattery.getWidth() * (BleManager.getInstance(PlugPiuTimer.this.mContext).getmConnBleDevice().getBattery() / 10.0d));
                    PlugPiuTimer plugPiuTimer = PlugPiuTimer.this;
                    if (width >= 0) {
                        plugPiuTimer.mImgBattery.setVisibility(0);
                    } else {
                        plugPiuTimer.mImgBattery.setVisibility(8);
                    }
                    layoutParams.width = width;
                    PlugPiuTimer.this.mImgBattery.setLayoutParams(layoutParams);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void pausePiuTask() {
        LogUtil.e("piuTimer广播：pausePiuTask");
        AnimUtil.pauseValueAnimation();
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        intentFilter.addAction(Constans.RECEIVER_PIU_TIMER_ACTION);
        AppUtil.registerReceiver(this.mContext, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void resumePiuTask() {
        LogUtil.e("piuTimer广播：resumePiuTask");
        AnimUtil.resumeValueAnimation();
    }

    private void setListener() {
        this.mImgPiuTime.setOnClickListener(this);
        this.mLayoutPiuBtn.setOnTouchListener(this.piuOnTouchListener);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void startPiuTask() {
        LogUtil.e("piuTimer广播：startPiuTask isPiuTimerPlay：" + this.isPiuTimerPlay);
        PiuTimerModel piuTimerModel = this.mPiuTimerModel;
        if (piuTimerModel == null || this.isPiuTimerPlay != 1) {
            return;
        }
        if (piuTimerModel.getSecond() == 0) {
            this.mHandler.sendEmptyMessageDelayed(18, 100L);
            PiuTimerManager.getInstance(this.mContext).nextTimer();
        } else {
            this.mProgressPiu.setMax(this.mPiuTimerModel.getSecond());
            AnimUtil.valueOfFloatAnimation(0.0f, this.mPiuTimerModel.getSecond(), new AnimUtil.onValueCallback() { // from class: com.hismith.hismithapp.view.plug.PlugPiuTimer.8
                @Override // com.hismith.hismithapp.utils.AnimUtil.onValueCallback
                public void onCancel() {
                    LogUtil.e("piu onCancel");
                }

                @Override // com.hismith.hismithapp.utils.AnimUtil.onValueCallback
                public void onEnd() {
                    LogUtil.e("piu end");
                    PlugPiuTimer.this.mProgressPiu.setProgress(0.0f);
                    if (PlugPiuTimer.this.mHandler.hasMessages(17)) {
                        PlugPiuTimer.this.mHandler.removeMessages(17);
                    }
                    PlugPiuTimer.this.mHandler.sendEmptyMessageDelayed(17, 100L);
                }

                @Override // com.hismith.hismithapp.utils.AnimUtil.onValueCallback
                public void onStart() {
                    LogUtil.e("piu onStart");
                    PlugPiuTimer.this.lastDown = 0;
                    PiuTimerManager.getInstance(PlugPiuTimer.this.mContext).setProgress(PlugPiuTimer.this.mPiuTimerModel.getSecond());
                    if (PlugPiuTimer.this.mHandler.hasMessages(17)) {
                        PlugPiuTimer.this.mHandler.removeMessages(17);
                    }
                }

                @Override // com.hismith.hismithapp.utils.AnimUtil.onValueCallback
                public void onValue(float f) {
                    PlugPiuTimer.this.mProgressPiu.setProgress(f);
                    int round = Math.round(PlugPiuTimer.this.mPiuTimerModel.getSecond() - f);
                    if (round != PlugPiuTimer.this.lastDown) {
                        PiuTimerManager.getInstance(PlugPiuTimer.this.mContext).setProgress(round);
                        PlugPiuTimer.this.mTxtPiuTime.setText(round + "''");
                        if (PlugPiuTimer.this.mPiuTimerDialog != null) {
                            PlugPiuTimer.this.mPiuTimerDialog.downNotifyAdapter(round);
                        }
                    }
                    if (f == PlugPiuTimer.this.mPiuTimerModel.getSecond()) {
                        LogUtil.e("piuTimer喷射!");
                        BleManager.getInstance(PlugPiuTimer.this.mContext).writeOpenPiu();
                        PiuTimerManager.getInstance(PlugPiuTimer.this.mContext).nextTimer();
                    }
                    PlugPiuTimer.this.lastDown = round;
                }
            });
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void stopPiuTask() {
        LogUtil.e("piuTimer广播：stopPiuTask");
        AnimUtil.cancelValueAnimation();
    }

    private void unregisterReceiver() {
        try {
            this.mContext.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        LogUtil.e("initFragment piuTimer lazyLoad()");
        init();
        setListener();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (R.id.img_piu_time == view.getId()) {
            PiuTimerDialog piuTimerDialog = this.mPiuTimerDialog;
            if (piuTimerDialog == null) {
                piuTimerDialog = new PiuTimerDialog(this.mActivity, this.mDialogHeightPercent);
                this.mPiuTimerDialog = piuTimerDialog;
            }
            piuTimerDialog.showDialog();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mContext = getContext();
        this.mActivity = getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.plug_piu_timer, (ViewGroup) null);
        this.rootView = inflate;
        this.mLayoutBattery = (ConstraintLayout) inflate.findViewById(R.id.layout_battery);
        this.mViewBattery = this.rootView.findViewById(R.id.view_battery);
        this.mImgBattery = (ImageView) this.rootView.findViewById(R.id.img_battery);
        this.mGroupPiu = (Group) this.rootView.findViewById(R.id.group_piu);
        this.mImgPiuTime = (ImageView) this.rootView.findViewById(R.id.img_piu_time);
        this.mTxtPiuTime = (TextView) this.rootView.findViewById(R.id.txt_piu_time);
        this.mProgressPiu = (GradientRoundProgress) this.rootView.findViewById(R.id.iplay_progress_piu);
        this.mLayoutPiuBtn = (FrameLayout) this.rootView.findViewById(R.id.iplay_layout_piu_btn);
        this.mImgPiu = (ImageView) this.rootView.findViewById(R.id.iplay_img_piu);
        LanguageUtil.setText(this.mContext, this.mTxtPiuTime, "spray");
        LogUtil.e("initFragment piuTimer onCreateView()");
        return this.rootView;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        PiuTimerManager.getInstance(this.mContext).clear();
        stopPiuTask();
        unregisterReceiver();
    }
}
