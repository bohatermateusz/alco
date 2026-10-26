package com.hismith.hismithapp.manager;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.graphics.Point;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.BounceInterpolator;
import android.view.animation.ScaleAnimation;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
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
public class RemoteFloatingManager implements SocketManager.OnSocketNotifyListener, BleManager.OnNotifyChangeListener_v2 {
    private static final int WHAT_CLIMAX = 3;
    private static final int WHAT_HIDE_HEART = 4;
    private static final int WHAT_MODE = 2;
    private static final int WHAT_MUSIC = 6;
    private static final int WHAT_SPEED = 1;
    private static final int WHAT_VIBRATE = 5;
    private static RemoteFloatingManager instance;
    private Callback callback;
    private final Context mContext;
    private ImageView mImgBg;
    private ImageView mImgHeart;
    private ImageView mImgMode;
    private FrameLayout mLayout;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private TextView mTxtSpeed;
    private View mView;
    private ImageView popup_float_mode_bg;
    private final Gson gson = new Gson();
    private int limitMax = 100;
    private int mSpeed = 0;
    private int mMode = 0;
    private int mVibreate = 0;
    private int mMusic = 0;
    private int mClimax = 0;
    private final boolean isReversal = false;
    private final Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.manager.RemoteFloatingManager.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                RemoteFloatingManager.this.mSpeed = message.arg1;
                RemoteFloatingManager.this.mMode = 0;
                if (RemoteFloatingManager.this.popup_float_mode_bg != null) {
                    RemoteFloatingManager.this.popup_float_mode_bg.setVisibility(4);
                }
                if (RemoteFloatingManager.this.mImgMode != null) {
                    RemoteFloatingManager.this.mImgMode.setVisibility(4);
                }
                if (RemoteFloatingManager.this.mTxtSpeed != null) {
                    RemoteFloatingManager.this.mTxtSpeed.setVisibility(0);
                    RemoteFloatingManager.this.mTxtSpeed.setText(RemoteFloatingManager.this.mSpeed + "");
                }
            } else if (i != 2) {
                if (i == 3) {
                    RemoteFloatingManager.this.mClimax = 1;
                    if (RemoteFloatingManager.this.callback != null) {
                        RemoteFloatingManager.this.callback.onShowHeart();
                    }
                    if (RemoteFloatingManager.this.mImgHeart != null) {
                        RemoteFloatingManager remoteFloatingManager = RemoteFloatingManager.this;
                        remoteFloatingManager.startHeart(remoteFloatingManager.mImgHeart);
                    }
                } else if (i == 4) {
                    RemoteFloatingManager.this.mClimax = 0;
                    if (RemoteFloatingManager.this.callback != null) {
                        RemoteFloatingManager.this.callback.onHideHeart();
                    }
                    if (RemoteFloatingManager.this.mImgHeart != null) {
                        RemoteFloatingManager remoteFloatingManager2 = RemoteFloatingManager.this;
                        remoteFloatingManager2.stopHeart(remoteFloatingManager2.mImgHeart);
                    }
                }
            } else if (ProductUtil.hasMode(0, 0)) {
                RemoteFloatingManager.this.mMode = message.arg1;
                if (RemoteFloatingManager.this.mImgMode != null) {
                    int i2 = RemoteFloatingManager.this.mMode;
                    RemoteFloatingManager remoteFloatingManager3 = RemoteFloatingManager.this;
                    if (i2 <= 0) {
                        if (remoteFloatingManager3.mTxtSpeed != null) {
                            RemoteFloatingManager.this.mTxtSpeed.setVisibility(0);
                        }
                        if (RemoteFloatingManager.this.mImgMode != null) {
                            RemoteFloatingManager.this.mImgMode.setVisibility(4);
                        }
                        if (RemoteFloatingManager.this.popup_float_mode_bg != null) {
                            RemoteFloatingManager.this.popup_float_mode_bg.setVisibility(4);
                        }
                    } else {
                        if (remoteFloatingManager3.mTxtSpeed != null) {
                            RemoteFloatingManager.this.mTxtSpeed.setVisibility(4);
                        }
                        if (RemoteFloatingManager.this.mImgMode != null) {
                            RemoteFloatingManager.this.mImgMode.setVisibility(0);
                        }
                        if (RemoteFloatingManager.this.popup_float_mode_bg != null) {
                            RemoteFloatingManager.this.popup_float_mode_bg.setVisibility(0);
                        }
                    }
                    List asList = Arrays.asList(ProductUtil.getProductMode(0));
                    int i3 = RemoteFloatingManager.this.mMode;
                    RemoteFloatingManager remoteFloatingManager4 = RemoteFloatingManager.this;
                    if (i3 <= 0) {
                        if (remoteFloatingManager4.mImgMode != null) {
                            Glide.with(MyApplication.getApplication()).load(Integer.valueOf(R.mipmap.ic_iplay_close_mode_2)).placeholder(R.mipmap.ic_iplay_def_button).into(RemoteFloatingManager.this.mImgMode);
                        }
                    } else if (remoteFloatingManager4.mMode <= asList.size() && RemoteFloatingManager.this.mImgMode != null) {
                        Glide.with(MyApplication.getApplication()).load(((ProductModel.Mode) asList.get(RemoteFloatingManager.this.mMode - 1)).getmIcon()).placeholder(R.mipmap.ic_iplay_def_button).into(RemoteFloatingManager.this.mImgMode);
                    }
                }
            }
            return false;
        }
    });
    private final PermissionListener mPermissionListener = new PermissionListener() { // from class: com.hismith.hismithapp.manager.RemoteFloatingManager.3
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
    private final ViewStateListener mViewStateListener = new ViewStateListener() { // from class: com.hismith.hismithapp.manager.RemoteFloatingManager.4
        @Override // com.yhao.floatwindow.ViewStateListener
        public void onBackToDesktop() {
            LogUtil.e("onBackToDesktop()");
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onDismiss() {
            LogUtil.e("onDismiss()");
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onHide() {
            LogUtil.e("onHide()");
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onMoveAnimEnd() {
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onMoveAnimStart() {
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onPositionUpdate(int i, int i2) {
            Point screenSize = MeasureUtil.getScreenSize(RemoteFloatingManager.this.mContext);
            if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) != null) {
                if (i2 < 0) {
                    FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).updateY(0);
                } else if (i2 > ((screenSize.y - MeasureUtil.dip2px(RemoteFloatingManager.this.mContext, 115.0f)) - MeasureUtil.getStatusBarHeight(RemoteFloatingManager.this.mContext)) - MeasureUtil.dip2px(RemoteFloatingManager.this.mContext, 6.0f)) {
                    FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).updateY(RemoteFloatingManager.this.lastY);
                    return;
                }
                RemoteFloatingManager.this.lastY = i2;
            }
        }

        @Override // com.yhao.floatwindow.ViewStateListener
        public void onShow() {
            LogUtil.e("onShow()");
        }
    };

    public interface Callback {
        void OnNotify(String str);

        void OnOpen();

        void onClose();

        void onCodeNull();

        void onHideHeart();

        void onNotifyChange(String str, String str2, String str3);

        void onShowHeart();
    }

    public RemoteFloatingManager(Context context) {
        this.mContext = context;
    }

    public static RemoteFloatingManager getInstance() {
        RemoteFloatingManager remoteFloatingManager;
        synchronized (RemoteFloatingManager.class) {
            remoteFloatingManager = instance;
            if (remoteFloatingManager == null) {
                remoteFloatingManager = new RemoteFloatingManager(MyApplication.getApplication());
                instance = remoteFloatingManager;
            }
        }
        return remoteFloatingManager;
    }

    private void initBleAAData(String str, String str2) {
        str.hashCode();
        int i = 0;
        switch (str) {
            case "03":
                this.mHandler.sendEmptyMessage(4);
                try {
                    SocketManager.getInstance().setSpeed(Integer.valueOf(str2, 16).intValue());
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                    break;
                }
            case "04":
                break;
            case "05":
                this.mHandler.sendEmptyMessage(4);
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    SocketManager.getInstance().setMode(intValue);
                    Message message = new Message();
                    message.what = 2;
                    message.arg1 = intValue;
                    this.mHandler.sendMessage(message);
                    return;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "06":
                try {
                    if (MyApplication.getApplication().getmProductModel() != null) {
                        if (MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                            this.mHandler.sendEmptyMessage(3);
                            return;
                        }
                        if (Constans.DEVICE_TYPE_SM2.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
                            int intValue2 = Integer.valueOf(str2, 16).intValue();
                            SocketManager.getInstance().setVibrateMode(intValue2);
                            Message message2 = new Message();
                            message2.what = 5;
                            message2.obj = Integer.valueOf(intValue2);
                            this.mHandler.sendMessage(message2);
                            return;
                        }
                        return;
                    }
                    return;
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
            default:
                return;
        }
        this.mHandler.sendEmptyMessage(4);
        try {
            int intValue3 = Integer.valueOf(str2, 16).intValue();
            if (intValue3 > 100) {
                i = 100;
            } else if (intValue3 >= 0) {
                i = intValue3;
            }
            Message message3 = new Message();
            message3.what = 1;
            message3.arg1 = i;
            this.mHandler.sendMessage(message3);
        } catch (Exception e4) {
            e4.printStackTrace();
        }
    }

    private void initBleCCData(String str, String str2) {
        int i;
        str.hashCode();
        i = 0;
        switch (str) {
            case "03":
                this.mHandler.sendEmptyMessage(4);
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message = new Message();
                    message.what = 1;
                    message.arg1 = i;
                    this.mHandler.sendMessage(message);
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                    return;
                }
            case "04":
                this.mHandler.sendEmptyMessage(4);
                try {
                    int intValue2 = Integer.valueOf(str2, 16).intValue();
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
                    int intValue3 = Integer.valueOf(str2, 16).intValue();
                    Message message3 = new Message();
                    message3.what = 5;
                    message3.obj = Integer.valueOf(intValue3);
                    this.mHandler.sendMessage(message3);
                    break;
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void launchActivity(Context context) {
        Intent intent = new Intent("android.intent.action.MAIN");
        intent.addCategory("android.intent.category.LAUNCHER");
        intent.setFlags(270532608);
        intent.setComponent(new ComponentName(this.mContext.getPackageName(), "com.hismith.hismithapp.RemotePlayActivity"));
        context.startActivity(intent);
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
            if (remotePlayModel.getSuccess() == 1 && !"1".equals(remotePlayModel.getClientConnection())) {
                if ("1".equals(remotePlayModel.getOffline())) {
                    BleManager.getInstance(this.mContext).writeShutdown();
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
                    BleManager.getInstance(this.mContext).writeSpeed(speed);
                    return;
                }
                if ("setPosition".equals(remotePlayModel.getAction())) {
                    return;
                }
                if (!"setMode".equals(remotePlayModel.getAction())) {
                    if ("setClimax".equals(remotePlayModel.getAction()) && MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                        this.mHandler.sendEmptyMessage(3);
                        boolean equals = "1003".equals(MyApplication.getApplication().getDeviceCode());
                        Context context = this.mContext;
                        if (equals) {
                            BleManager.getInstance(context).writeOpenClimax();
                            return;
                        } else {
                            BleManager.getInstance(context).writeOpenClimax01();
                            return;
                        }
                    }
                    return;
                }
                int mode = remotePlayModel.getMode();
                if (!"modes".equals(remotePlayModel.getType())) {
                    if ("vibrations".equals(remotePlayModel.getType())) {
                        this.mVibreate = mode;
                        if (mode == 0) {
                            BleManager.getInstance(this.mContext).writeCloseVibrate();
                            return;
                        }
                        boolean hasMode = ProductUtil.hasMode(1, mode);
                        Context context2 = this.mContext;
                        if (hasMode) {
                            BleManager.getInstance(context2).writeData(ProductUtil.getProductMode(1)[this.mVibreate - 1].getmCommand());
                            return;
                        } else {
                            BleManager.getInstance(context2).writeVibrateMode(this.mVibreate);
                            return;
                        }
                    }
                    return;
                }
                Message message2 = new Message();
                message2.what = 2;
                message2.arg1 = mode;
                this.mHandler.sendMessage(message2);
                if (mode <= 0) {
                    BleManager.getInstance(this.mContext).writeSpeed(this.mSpeed);
                    return;
                }
                boolean hasMode2 = ProductUtil.hasMode(0, mode);
                Context context3 = this.mContext;
                if (hasMode2) {
                    BleManager.getInstance(context3).writeData(ProductUtil.getProductMode(0)[mode - 1].getmCommand());
                } else {
                    BleManager.getInstance(context3).writeSpeedMode(mode);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void destroyFloatWindow() {
        if (SocketManager.getInstance().getOnSocketNotifyListener() instanceof RemoteFloatingManager) {
            LogUtil.e("------------------清除回调------------------");
            SocketManager.getInstance().setOnSocketNotifyListener(null);
        }
        if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) != null) {
            FloatWindow.destroy(Constans.FLOAT_REMOTE_WINDOW);
        }
    }

    public int getLimitMax() {
        return this.limitMax;
    }

    public void initListener() {
        BleManager.getInstance(this.mContext).setOnNotifyChangeListener_v2(this);
        SocketManager.getInstance().setOnSocketNotifyListener(this);
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

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onFailure(Throwable th, Response response) {
        destroyFloatWindow();
        Callback callback = this.callback;
        if (callback != null) {
            callback.onClose();
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnNotifyChangeListener_v2
    public void onNotifyChange_v2(String str, String str2, String str3) {
        if ("AA".equals(str)) {
            initBleAAData(str2, str3);
        } else if ("CA".equals(str) || "CC".equals(str)) {
            initBleCCData(str2, str3);
        }
        Callback callback = this.callback;
        if (callback != null) {
            callback.onNotifyChange(str, str2, str3);
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onOpen() {
        Callback callback = this.callback;
        if (callback != null) {
            callback.OnOpen();
        }
    }

    public void setCallback(Callback callback) {
        this.callback = callback;
    }

    public void setLimitMax(int i) {
        this.limitMax = i;
    }

    public void showFloatWindow(int i, int i2, int i3, int i4, int i5, int i6) {
        this.limitMax = i4;
        this.mVibreate = i5;
        this.mMusic = i6;
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.popup_floating, (ViewGroup) null, false);
        this.mView = inflate;
        this.mImgBg = (ImageView) inflate.findViewById(R.id.popup_float_img_bg);
        this.mLayout = (FrameLayout) this.mView.findViewById(R.id.popup_float_layout);
        this.popup_float_mode_bg = (ImageView) this.mView.findViewById(R.id.popup_float_mode_bg);
        this.mImgMode = (ImageView) this.mView.findViewById(R.id.popup_float_img_mode);
        this.mTxtSpeed = (TextView) this.mView.findViewById(R.id.popup_float_txt_speed);
        this.mImgHeart = (ImageView) this.mView.findViewById(R.id.popup_float_img_heart);
        this.popup_float_mode_bg.setVisibility(4);
        this.mImgMode.setVisibility(4);
        this.mImgHeart.setVisibility(4);
        this.mView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.manager.RemoteFloatingManager.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                RemoteFloatingManager remoteFloatingManager = RemoteFloatingManager.this;
                remoteFloatingManager.launchActivity(remoteFloatingManager.mContext);
                RemoteFloatingManager.this.destroyFloatWindow();
            }
        });
        if (i == 0) {
            this.mTxtSpeed.setText("0");
        } else if (i == 1) {
            Message message = new Message();
            message.what = 1;
            message.arg1 = i2;
            this.mHandler.sendMessage(message);
        } else if (i == 2) {
            Message message2 = new Message();
            message2.what = 2;
            message2.arg1 = i3;
            this.mHandler.sendMessage(message2);
        } else if (i == 3) {
            this.mHandler.sendEmptyMessage(3);
        }
        Point screenSize = MeasureUtil.getScreenSize(this.mContext);
        int dip2px = MeasureUtil.dip2px(this.mContext, 70.0f);
        int dip2px2 = MeasureUtil.dip2px(this.mContext, 6.0f);
        try {
            if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) != null) {
                FloatWindow.destroy(Constans.FLOAT_REMOTE_WINDOW);
            }
            FloatWindow.with(this.mContext).setView(this.mView).setTag(Constans.FLOAT_REMOTE_WINDOW).setWidth(dip2px).setHeight(MeasureUtil.dip2px(this.mContext, 100.0f)).setX(screenSize.x - (dip2px + dip2px2)).setY(1, 0.3f).setMoveType(3, dip2px2, dip2px2).setMoveStyle(500L, new BounceInterpolator()).setFilter(false, RemotePlayActivity.class).setViewStateListener(this.mViewStateListener).setPermissionListener(this.mPermissionListener).setDesktopShow(true).build();
            FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
