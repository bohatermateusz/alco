package com.hismith.hismithapp.service;

import android.app.Service;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.graphics.Point;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Binder;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.BounceInterpolator;
import android.view.animation.LinearInterpolator;
import android.view.animation.RotateAnimation;
import android.view.animation.ScaleAnimation;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.media3.common.C;
import com.bumptech.glide.Glide;
import com.google.gson.Gson;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.RemotePlayActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.RemotePlayModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.yhao.floatwindow.FloatWindow;
import com.yhao.floatwindow.PermissionListener;
import com.yhao.floatwindow.ViewStateListener;
import java.util.Arrays;
import java.util.List;
import okhttp3.Response;

/* loaded from: classes3.dex */
public class FloatingWindowService extends Service implements SocketManager.OnSocketNotifyListener, BleManager.OnNotifyChangeListener_v2, SensorEventListener {
    private static final int WHAT_CLIMAX = 3;
    private static final int WHAT_HIDE_HEART = 4;
    private static final int WHAT_MODE = 2;
    private static final int WHAT_MUSIC = 6;
    private static final int WHAT_SPEED = 1;
    private static final int WHAT_VIBRATE = 5;
    private Callback callback;
    private FloatWindow floatWindow;
    public Gson gson;
    private ImageView mImgBg;
    private ImageView mImgHeart;
    private ImageView mImgMode;
    private FrameLayout mLayout;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private TextView mTxtSpeed;
    private View mView;
    private ImageView popup_float_mode_bg;
    private int limitMax = 100;
    private int mSpeed = 0;
    private int mMode = 0;
    private int mVibreate = 0;
    private int mMusic = 0;
    private int mClimax = 0;
    private boolean isReversal = false;
    private FloatingBinder floatingBinder = new FloatingBinder();
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.service.FloatingWindowService.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                FloatingWindowService.this.mSpeed = message.arg1;
                FloatingWindowService.this.mMode = 0;
                if (FloatingWindowService.this.popup_float_mode_bg != null) {
                    FloatingWindowService.this.popup_float_mode_bg.setVisibility(4);
                }
                if (FloatingWindowService.this.mImgMode != null) {
                    FloatingWindowService.this.mImgMode.setVisibility(4);
                }
                if (FloatingWindowService.this.mTxtSpeed != null) {
                    FloatingWindowService.this.mTxtSpeed.setVisibility(0);
                    FloatingWindowService.this.mTxtSpeed.setText(FloatingWindowService.this.mSpeed + "");
                }
            } else if (i != 2) {
                if (i == 3) {
                    FloatingWindowService.this.mClimax = 1;
                    if (FloatingWindowService.this.callback != null) {
                        FloatingWindowService.this.callback.onShowHeart();
                    }
                    if (FloatingWindowService.this.mImgHeart != null) {
                        FloatingWindowService floatingWindowService = FloatingWindowService.this;
                        floatingWindowService.startHeart(floatingWindowService.mImgHeart);
                    }
                } else if (i == 4) {
                    FloatingWindowService.this.mClimax = 0;
                    if (FloatingWindowService.this.callback != null) {
                        FloatingWindowService.this.callback.onHideHeart();
                    }
                    if (FloatingWindowService.this.mImgHeart != null) {
                        FloatingWindowService floatingWindowService2 = FloatingWindowService.this;
                        floatingWindowService2.stopHeart(floatingWindowService2.mImgHeart);
                    }
                }
            } else if (ProductUtil.hasMode(0)) {
                FloatingWindowService.this.mMode = message.arg1;
                if (FloatingWindowService.this.mImgMode != null) {
                    int i2 = FloatingWindowService.this.mMode;
                    FloatingWindowService floatingWindowService3 = FloatingWindowService.this;
                    if (i2 <= 0) {
                        if (floatingWindowService3.mTxtSpeed != null) {
                            FloatingWindowService.this.mTxtSpeed.setVisibility(0);
                        }
                        if (FloatingWindowService.this.mImgMode != null) {
                            FloatingWindowService.this.mImgMode.setVisibility(4);
                        }
                        if (FloatingWindowService.this.popup_float_mode_bg != null) {
                            FloatingWindowService.this.popup_float_mode_bg.setVisibility(4);
                        }
                    } else {
                        if (floatingWindowService3.mTxtSpeed != null) {
                            FloatingWindowService.this.mTxtSpeed.setVisibility(4);
                        }
                        if (FloatingWindowService.this.mImgMode != null) {
                            FloatingWindowService.this.mImgMode.setVisibility(0);
                        }
                        if (FloatingWindowService.this.popup_float_mode_bg != null) {
                            FloatingWindowService.this.popup_float_mode_bg.setVisibility(0);
                        }
                    }
                    List asList = Arrays.asList(ProductUtil.getProductMode(0));
                    int i3 = FloatingWindowService.this.mMode;
                    FloatingWindowService floatingWindowService4 = FloatingWindowService.this;
                    if (i3 <= 0) {
                        if (floatingWindowService4.mImgMode != null) {
                            Glide.with(MyApplication.getApplication()).load(Integer.valueOf(R.mipmap.ic_iplay_close_mode_2)).placeholder(R.mipmap.ic_iplay_def_button).into(FloatingWindowService.this.mImgMode);
                        }
                    } else if (floatingWindowService4.mMode <= asList.size() && FloatingWindowService.this.mImgMode != null) {
                        Glide.with(MyApplication.getApplication()).load(((ProductModel.Mode) asList.get(FloatingWindowService.this.mMode - 1)).getmIcon()).placeholder(R.mipmap.ic_iplay_def_button).into(FloatingWindowService.this.mImgMode);
                    }
                }
            }
            return false;
        }
    });
    private long reversalTime = 0;
    private int reversalCount = 0;
    private int recoverCount = 0;
    private PermissionListener mPermissionListener = new PermissionListener() { // from class: com.hismith.hismithapp.service.FloatingWindowService.6
        @Override // com.yhao.floatwindow.PermissionListener
        public void onFail() {
            LogUtil.e("onFail");
        }

        @Override // com.yhao.floatwindow.PermissionListener
        public void onSuccess() {
            LogUtil.e("onSuccess");
        }
    };
    private int lastY = 0;
    private ViewStateListener mViewStateListener = new ViewStateListener() { // from class: com.hismith.hismithapp.service.FloatingWindowService.7
        @Override // com.yhao.floatwindow.ViewStateListener
        public void onBackToDesktop() {
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onDismiss() {
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onHide() {
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onMoveAnimEnd() {
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onMoveAnimStart() {
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onPositionUpdate(int i, int i2) {
            Point screenSize = MeasureUtil.getScreenSize(FloatingWindowService.this);
            if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) != null) {
                if (i2 < 0) {
                    FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).updateY(0);
                } else if (i2 > ((screenSize.y - MeasureUtil.dip2px(FloatingWindowService.this, 115.0f)) - MeasureUtil.getStatusBarHeight(FloatingWindowService.this)) - MeasureUtil.dip2px(FloatingWindowService.this, 6.0f)) {
                    FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).updateY(FloatingWindowService.this.lastY);
                    return;
                }
                FloatingWindowService.this.lastY = i2;
            }
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onShow() {
        }
    };

    public interface Callback {
        void OnNotify(String str);

        void OnOpen();

        void onClose();

        void onCodeNull();

        void onHideHeart();

        void onNotifyChange(String str, String str2);

        void onShowHeart();
    }

    public class FloatingBinder extends Binder {
        public FloatingBinder() {
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void launchActivity() {
            Intent intent = new Intent("android.intent.action.MAIN");
            intent.addCategory("android.intent.category.LAUNCHER");
            intent.setFlags(270532608);
            intent.setComponent(new ComponentName(FloatingWindowService.this.getPackageName(), "com.hismith.hismithapp.RemotePlayActivity"));
            FloatingWindowService.this.startActivity(intent);
        }

        private void rotate(View view) {
            RotateAnimation rotateAnimation = new RotateAnimation(0.0f, 360.0f, 1, 0.5f, 1, 0.5f);
            rotateAnimation.setInterpolator(new LinearInterpolator());
            rotateAnimation.setDuration(C.DEFAULT_MAX_SEEK_TO_PREVIOUS_POSITION_MS);
            rotateAnimation.setFillAfter(true);
            rotateAnimation.setRepeatCount(-1);
            view.startAnimation(rotateAnimation);
        }

        public void destroyFloatWindow() {
            LogUtil.e("destroyFloatWindow()");
            if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) != null) {
                FloatWindow.destroy(Constans.FLOAT_REMOTE_WINDOW);
            }
        }

        public FloatingWindowService getService() {
            if ("1".equals(SocketManager.getInstance().getClientConnection())) {
                FloatingWindowService.this.initClientConnection();
            }
            return FloatingWindowService.this;
        }

        public void showFloatWindow(Context context, int i, int i2, int i3, int i4, int i5, int i6) {
            FloatingWindowService.this.limitMax = i4;
            FloatingWindowService.this.mVibreate = i5;
            FloatingWindowService.this.mMusic = i6;
            FloatingWindowService.this.mView = LayoutInflater.from(context).inflate(R.layout.popup_floating, (ViewGroup) null, false);
            FloatingWindowService floatingWindowService = FloatingWindowService.this;
            floatingWindowService.mImgBg = (ImageView) floatingWindowService.mView.findViewById(R.id.popup_float_img_bg);
            FloatingWindowService floatingWindowService2 = FloatingWindowService.this;
            floatingWindowService2.mLayout = (FrameLayout) floatingWindowService2.mView.findViewById(R.id.popup_float_layout);
            FloatingWindowService floatingWindowService3 = FloatingWindowService.this;
            floatingWindowService3.popup_float_mode_bg = (ImageView) floatingWindowService3.mView.findViewById(R.id.popup_float_mode_bg);
            FloatingWindowService floatingWindowService4 = FloatingWindowService.this;
            floatingWindowService4.mImgMode = (ImageView) floatingWindowService4.mView.findViewById(R.id.popup_float_img_mode);
            FloatingWindowService floatingWindowService5 = FloatingWindowService.this;
            floatingWindowService5.mTxtSpeed = (TextView) floatingWindowService5.mView.findViewById(R.id.popup_float_txt_speed);
            FloatingWindowService floatingWindowService6 = FloatingWindowService.this;
            floatingWindowService6.mImgHeart = (ImageView) floatingWindowService6.mView.findViewById(R.id.popup_float_img_heart);
            FloatingWindowService.this.popup_float_mode_bg.setVisibility(4);
            FloatingWindowService.this.mImgMode.setVisibility(4);
            FloatingWindowService.this.mImgHeart.setVisibility(4);
            FloatingWindowService.this.mView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.service.FloatingWindowService.FloatingBinder.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    FloatingBinder.this.launchActivity();
                    FloatingBinder.this.destroyFloatWindow();
                }
            });
            if (i == 0) {
                FloatingWindowService.this.mTxtSpeed.setText("0");
            } else if (i == 1) {
                Message message = new Message();
                message.what = 1;
                message.arg1 = i2;
                FloatingWindowService.this.mHandler.sendMessage(message);
            } else if (i == 2) {
                Message message2 = new Message();
                message2.what = 2;
                message2.arg1 = i3;
                FloatingWindowService.this.mHandler.sendMessage(message2);
            } else if (i == 3) {
                FloatingWindowService.this.mHandler.sendEmptyMessage(3);
            }
            Point screenSize = MeasureUtil.getScreenSize(context);
            int dip2px = MeasureUtil.dip2px(context, 70.0f);
            int dip2px2 = MeasureUtil.dip2px(context, -6.0f);
            try {
                if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) != null) {
                    FloatWindow.destroy(Constans.FLOAT_REMOTE_WINDOW);
                }
                FloatWindow.with(FloatingWindowService.this.getApplicationContext()).setView(FloatingWindowService.this.mView).setTag(Constans.FLOAT_REMOTE_WINDOW).setWidth(dip2px).setHeight(MeasureUtil.dip2px(context, 100.0f)).setX(screenSize.x - (dip2px + dip2px2)).setY(1, 0.3f).setMoveType(3, dip2px2, dip2px2).setMoveStyle(500L, new BounceInterpolator()).setFilter(false, RemotePlayActivity.class).setViewStateListener(FloatingWindowService.this.mViewStateListener).setPermissionListener(FloatingWindowService.this.mPermissionListener).setDesktopShow(true).build();
                FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).show();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initClientConnection() {
        BleManager.getInstance(this).setOnNotifyChangeListener_v2(this);
    }

    private void initSensor() {
        SensorManager sensorManager = (SensorManager) getSystemService("sensor");
        this.mSensorManager = sensorManager;
        Sensor defaultSensor = sensorManager.getDefaultSensor(1);
        this.mSensor = defaultSensor;
        if (defaultSensor != null) {
            this.mSensorManager.registerListener(this, defaultSensor, 2);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void startHeart(ImageView imageView) {
        imageView.setVisibility(0);
        ScaleAnimation scaleAnimation = new ScaleAnimation(1.0f, 0.7f, 1.0f, 0.7f, 1, 0.5f, 1, 0.5f);
        scaleAnimation.setDuration(250L);
        scaleAnimation.setFillAfter(true);
        scaleAnimation.setRepeatCount(-1);
        scaleAnimation.setRepeatMode(2);
        imageView.startAnimation(scaleAnimation);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void stopHeart(ImageView imageView) {
        if (imageView.getVisibility() == 0) {
            imageView.clearAnimation();
            imageView.setVisibility(8);
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void OnNotify(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        LogUtil.e("OnNotify : 接收:" + str);
        Callback callback = this.callback;
        if (callback != null) {
            callback.OnNotify(str);
        }
        try {
            RemotePlayModel remotePlayModel = (RemotePlayModel) this.gson.fromJson(str, RemotePlayModel.class);
            if (remotePlayModel.getSuccess() == 1) {
                if ("1".equals(remotePlayModel.getClientConnection())) {
                    initClientConnection();
                    return;
                }
                if ("1".equals(remotePlayModel.getOffline())) {
                    BleManager.getInstance(this).writeShutdown();
                    return;
                }
                if ("setSpeed".equals(remotePlayModel.getAction())) {
                    int speed = remotePlayModel.getSpeed();
                    int i = this.limitMax;
                    if (speed > i) {
                        speed = i;
                    }
                    Message message = new Message();
                    message.what = 1;
                    message.arg1 = speed;
                    this.mHandler.sendMessage(message);
                    if (this.isReversal) {
                        return;
                    }
                    BleManager.getInstance(this).writeSpeed(speed);
                    return;
                }
                if ("setPosition".equals(remotePlayModel.getAction())) {
                    return;
                }
                if (!"setMode".equals(remotePlayModel.getAction())) {
                    if ("setClimax".equals(remotePlayModel.getAction()) && MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                        this.mHandler.sendEmptyMessage(3);
                        if (this.isReversal) {
                            return;
                        }
                        if ("1003".equals(MyApplication.getApplication().getDeviceCode())) {
                            BleManager.getInstance(this).writeOpenClimax();
                            return;
                        } else {
                            BleManager.getInstance(this).writeOpenClimax01();
                            return;
                        }
                    }
                    return;
                }
                int mode = remotePlayModel.getMode();
                if (!"modes".equals(remotePlayModel.getType())) {
                    if ("vibrations".equals(remotePlayModel.getType())) {
                        this.mVibreate = mode;
                        if (this.isReversal) {
                            return;
                        }
                        if (mode == 0) {
                            BleManager.getInstance(this).writeCloseVibrate();
                            return;
                        } else if (ProductUtil.hasMode(1, mode)) {
                            BleManager.getInstance(this).writeData(ProductUtil.getProductMode(1)[this.mVibreate - 1].getmCommand());
                            return;
                        } else {
                            BleManager.getInstance(this).writeVibrateMode(this.mVibreate);
                            return;
                        }
                    }
                    return;
                }
                Message message2 = new Message();
                message2.what = 2;
                message2.arg1 = mode;
                this.mHandler.sendMessage(message2);
                if (this.isReversal) {
                    return;
                }
                if (mode <= 0) {
                    BleManager.getInstance(this).writeSpeed(this.mSpeed);
                } else if (ProductUtil.hasMode(0, mode)) {
                    BleManager.getInstance(this).writeData(ProductUtil.getProductMode(0)[mode - 1].getmCommand());
                } else {
                    BleManager.getInstance(this).writeSpeedMode(mode);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // android.hardware.SensorEventListener
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    @Override // android.app.Service
    public IBinder onBind(Intent intent) {
        SocketManager.getInstance().setOnSocketNotifyListener(this);
        return this.floatingBinder;
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onClosed(int i, String str) {
        Callback callback = this.callback;
        if (callback != null) {
            callback.onClose();
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onCodeNull() {
        Callback callback = this.callback;
        if (callback != null) {
            callback.onCodeNull();
        }
    }

    @Override // android.app.Service
    public void onCreate() {
        super.onCreate();
        this.gson = new Gson();
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onFailure(Throwable th, Response response) {
        Callback callback = this.callback;
        if (callback != null) {
            callback.onClose();
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnNotifyChangeListener_v2
    public void onNotifyChange_v2(String str, String str2, String str3) {
        int i;
        Callback callback = this.callback;
        if (callback != null) {
            callback.onNotifyChange(str2, str3);
        }
        str2.hashCode();
        i = 0;
        switch (str2) {
            case "03":
            case "04":
                this.mHandler.sendEmptyMessage(4);
                try {
                    int intValue = Integer.valueOf(str3, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    SocketManager.getInstance().setSpeed(i);
                    Message message = new Message();
                    message.what = 1;
                    message.arg1 = i;
                    this.mHandler.sendMessage(message);
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                    return;
                }
            case "05":
                this.mHandler.sendEmptyMessage(4);
                try {
                    int intValue2 = Integer.valueOf(str3, 16).intValue();
                    SocketManager.getInstance().setMode(intValue2);
                    Message message2 = new Message();
                    message2.what = 2;
                    message2.arg1 = intValue2;
                    this.mHandler.sendMessage(message2);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "06":
                try {
                    if (MyApplication.getApplication().getmProductModel() != null) {
                        if (MyApplication.getApplication().getmProductModel().getClimax() != 1) {
                            if (ProductUtil.isSM2Type()) {
                                int intValue3 = Integer.valueOf(str3, 16).intValue();
                                SocketManager.getInstance().setVibrateMode(intValue3);
                                Message message3 = new Message();
                                message3.what = 5;
                                message3.obj = Integer.valueOf(intValue3);
                                this.mHandler.sendMessage(message3);
                                break;
                            }
                        } else {
                            this.mHandler.sendEmptyMessage(3);
                            break;
                        }
                    }
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
                break;
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onOpen() {
        Callback callback = this.callback;
        if (callback != null) {
            callback.OnOpen();
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
            if (f3 <= -9.0f && f3 > -10.0f) {
                int i = this.reversalCount + 1;
                this.reversalCount = i;
                if (i >= 4) {
                    this.recoverCount = 0;
                    if (!this.isReversal) {
                        this.isReversal = true;
                        BleManager.getInstance(this).writeSpeed(0);
                        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.FloatingWindowService.2
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(FloatingWindowService.this).writeCloseVibrate();
                            }
                        }, 100L);
                        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.FloatingWindowService.3
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(FloatingWindowService.this).writeCloseVoice();
                            }
                        }, 200L);
                    }
                }
            } else if (f3 > -1.0f) {
                int i2 = this.recoverCount + 1;
                this.recoverCount = i2;
                if (i2 >= 4) {
                    this.reversalCount = 0;
                    if (this.isReversal) {
                        this.isReversal = false;
                        int i3 = this.mMode;
                        if (i3 <= 0) {
                            BleManager.getInstance(this).writeSpeed(this.mSpeed);
                        } else if (ProductUtil.hasMode(0, i3)) {
                            BleManager.getInstance(this).writeData(ProductUtil.getProductMode(0)[this.mMode - 1].getmCommand());
                        } else {
                            BleManager.getInstance(this).writeSpeedMode(this.mMode);
                        }
                        if (this.mVibreate > 0) {
                            new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.FloatingWindowService.4
                                @Override // java.lang.Runnable
                                public void run() {
                                    boolean hasMode = ProductUtil.hasMode(1, FloatingWindowService.this.mVibreate);
                                    FloatingWindowService floatingWindowService = FloatingWindowService.this;
                                    if (hasMode) {
                                        BleManager.getInstance(floatingWindowService).writeData(ProductUtil.getProductMode(1)[FloatingWindowService.this.mVibreate - 1].getmCommand());
                                    } else {
                                        BleManager.getInstance(floatingWindowService).writeVibrateMode(FloatingWindowService.this.mVibreate);
                                    }
                                }
                            }, 100L);
                        }
                        if (this.mMusic > 0) {
                            new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.FloatingWindowService.5
                                @Override // java.lang.Runnable
                                public void run() {
                                    BleManager.getInstance(FloatingWindowService.this).writeOpenVoice();
                                }
                            }, 200L);
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

    @Override // android.app.Service
    public boolean onUnbind(Intent intent) {
        FloatWindow.destroy(Constans.FLOAT_REMOTE_WINDOW);
        BleManager.getInstance(this).writeShutdown();
        BleManager.getInstance(this).setOnNotifyChangeListener_v2(null);
        BleManager.getInstance(this).setOnConnectStateListener(null);
        return super.onUnbind(intent);
    }

    public void setCallback(Callback callback) {
        this.callback = callback;
    }

    public void setLimitMax(int i) {
        this.limitMax = i;
    }
}
