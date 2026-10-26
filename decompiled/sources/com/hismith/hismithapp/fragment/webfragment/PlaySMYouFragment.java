package com.hismith.hismithapp.fragment.webfragment;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.ValueCallback;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ImageView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.ConstraintSet;
import androidx.fragment.app.FragmentActivity;
import com.bumptech.glide.Glide;
import com.google.android.gms.common.util.ArrayUtils;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.fragment.PlaySMSFragment2;
import com.hismith.hismithapp.fragment.SocketPlayFragment;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.BleSpeedUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.ColorUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HudongModeLocationUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMYouFragment extends SocketPlayFragment implements View.OnClickListener, SensorEventListener {
    private BaseActivity baseActivity;
    private ImageView img_reversal;
    private ImageView img_speed_mode;
    private ImageView img_vibrate_mode;
    private ConstraintLayout layout_root;
    private ConstraintLayout layout_speed_mode;
    private ConstraintLayout layout_vibrate_mode;
    private BleSpeedUtil mBleSpeedUtil;
    private Gson mGson;
    private List<String> mModes;
    private SocketProductModel mProductModel;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.8
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (Constans.RECEIVER_SVG_DOWNLOAD.equals(intent.getAction())) {
                PlaySMYouFragment.this.initWebView();
            }
        }
    };
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private List<String> mVibrateModes;
    private PlayView_Four playView;
    private VerticalProgressBar progress_speed;
    private RecorderVolumeManager recorderVolumeManager;
    private CustomWebView web_device;

    /* JADX INFO: Access modifiers changed from: private */
    public SocketProductModel getBinderProduct() {
        BaseActivity baseActivity = this.baseActivity;
        return baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getBinder().getSocketProductModel() : baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getBinder().getSocketProductModel() : new SocketProductModel();
    }

    private void init() {
        this.mGson = new Gson();
        this.mModes = new ArrayList();
        this.mVibrateModes = new ArrayList();
        Bundle arguments = getArguments();
        if (arguments != null) {
            this.mProductModel = (SocketProductModel) arguments.getSerializable("ProductModel");
        }
        SocketProductModel socketProductModel = this.mProductModel;
        if (socketProductModel != null) {
            initProduct(socketProductModel);
        }
        this.recorderVolumeManager = new RecorderVolumeManager();
        initWebView();
        initSensor();
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleDataUI(SocketProductModel socketProductModel) {
        if (socketProductModel != null) {
            refreshSpeedUI(socketProductModel.getSpeed(), 0);
            refreshSpeedModeUI(socketProductModel.getMode());
            refreshVibrateModeUI(socketProductModel.getVibration());
        }
    }

    private void initProduct(SocketProductModel socketProductModel) {
        if (socketProductModel != null) {
            PlayView_Four playView_Four = this.playView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(socketProductModel.getSlide());
            }
            if (socketProductModel.getModesUrl() != null) {
                if (socketProductModel.getModesUrl().getModes() != null) {
                    this.mModes.addAll(ArrayUtils.toArrayList(socketProductModel.getModesUrl().getModes()));
                }
                if (socketProductModel.getModesUrl().getVibrations() != null) {
                    this.mVibrateModes.addAll(ArrayUtils.toArrayList(socketProductModel.getModesUrl().getVibrations()));
                }
            }
            int reversalFun = socketProductModel.getReversalFun();
            ImageView imageView = this.img_reversal;
            if (reversalFun != 1) {
                imageView.clearAnimation();
                this.img_reversal.setVisibility(8);
                return;
            }
            imageView.clearAnimation();
            this.img_reversal.setVisibility(0);
            int reversal = socketProductModel.getReversal();
            ImageView imageView2 = this.img_reversal;
            if (reversal == 1) {
                AnimUtil.rotate(imageView2, 1);
                this.img_reversal.setRotationY(180.0f);
            } else {
                AnimUtil.rotate(imageView2, 0);
                this.img_reversal.setRotationY(0.0f);
            }
        }
    }

    private void initSensor() {
        SensorManager sensorManager = (SensorManager) this.baseActivity.getSystemService("sensor");
        this.mSensorManager = sensorManager;
        this.mSensor = sensorManager.getDefaultSensor(1);
        this.mBleSpeedUtil = new BleSpeedUtil(this.baseActivity);
    }

    private void initView(View view) {
        this.layout_root = (ConstraintLayout) view.findViewById(R.id.layout_root);
        this.web_device = (CustomWebView) view.findViewById(R.id.web_device);
        this.playView = (PlayView_Four) view.findViewById(R.id.playView);
        this.progress_speed = (VerticalProgressBar) view.findViewById(R.id.progress_speed);
        this.img_reversal = (ImageView) view.findViewById(R.id.img_reversal);
        this.layout_speed_mode = (ConstraintLayout) view.findViewById(R.id.layout_speed_mode);
        this.img_speed_mode = (ImageView) view.findViewById(R.id.img_speed_mode);
        this.layout_vibrate_mode = (ConstraintLayout) view.findViewById(R.id.layout_vibrate_mode);
        this.img_vibrate_mode = (ImageView) view.findViewById(R.id.img_vibrate_mode);
        this.playView.setProgressBar(this.progress_speed);
        this.playView.setMy(false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initWebView() {
        String str;
        this.web_device.getSettings().setCacheMode(1);
        this.web_device.setBackgroundColor(0);
        this.web_device.setInitialScale(100);
        WebSettings settings = this.web_device.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setJavaScriptCanOpenWindowsAutomatically(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        SocketProductModel socketProductModel = this.mProductModel;
        if (socketProductModel != null) {
            String svgName = socketProductModel.getSvgName();
            HudongModeLocationUtil.initModePosition(svgName, this.layout_root, R.id.web_device, R.id.layout_speed_mode, R.id.layout_vibrate_mode, 1);
            if (this.mProductModel.getSvgLeftRight() == 1) {
                svgName = svgName + "_left";
            }
            if (CacheUtil.playSVGExists(this.baseActivity, svgName)) {
                str = "file:///" + new File(this.baseActivity.getFilesDir() + "/playSVG", svgName + ".svg").getPath();
            } else {
                str = "file:///android_asset/speed.svg";
                svgName = "speed.svg";
            }
            if (svgName.equals("speed.svg") || svgName.contains("02device") || svgName.contains("03device") || svgName.contains("04device") || svgName.contains("05device") || svgName.contains("06device") || svgName.contains("07device") || svgName.contains("09device")) {
                ConstraintSet constraintSet = new ConstraintSet();
                constraintSet.clone(this.layout_root);
                constraintSet.constrainPercentWidth(R.id.web_device, 0.15f);
                constraintSet.setDimensionRatio(R.id.web_device, "H,1:2.75");
                constraintSet.applyTo(this.layout_root);
            } else if (svgName.contains("08device")) {
                ConstraintSet constraintSet2 = new ConstraintSet();
                constraintSet2.clone(this.layout_root);
                constraintSet2.constrainPercentWidth(R.id.web_device, 0.34f);
                constraintSet2.setDimensionRatio(R.id.web_device, "H,1:0.788");
                constraintSet2.applyTo(this.layout_root);
            } else {
                ConstraintSet constraintSet3 = new ConstraintSet();
                constraintSet3.clone(this.layout_root);
                constraintSet3.constrainPercentWidth(R.id.web_device, 0.5f);
                constraintSet3.setDimensionRatio(R.id.web_device, "H,1:0.383");
                constraintSet3.applyTo(this.layout_root);
            }
        } else {
            str = "file:///android_asset/01device_left.svg";
        }
        if (TextUtils.isEmpty(str)) {
            LogUtil.e("webView uri:NULL");
        } else {
            LogUtil.e("webView uri:" + str);
            this.web_device.loadUrl(str);
        }
        this.web_device.setWebViewClient(new WebViewClient() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.1
            @Override // android.webkit.WebViewClient
            public void onPageFinished(WebView webView, String str2) {
                super.onPageFinished(webView, str2);
                PlaySMYouFragment.this.refreshColorStyleUI(R.color.app_style_color10, 1);
                PlaySMYouFragment.this.refreshColorStyleUI(R.color.app_style_color, 0);
                PlaySMYouFragment.this.refreshPositionStatus(0);
                PlaySMYouFragment playSMYouFragment = PlaySMYouFragment.this;
                playSMYouFragment.initBleDataUI(playSMYouFragment.mProductModel);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshColorStyleUI(int i, int i2) {
        this.web_device.evaluateJavascript("javascript:setColorStyle('" + ColorUtil.toHexEncoding(this.baseActivity, i) + "', " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.6
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setColorStyle Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionStatus(int i) {
        CustomWebView customWebView = this.web_device;
        if (customWebView != null) {
            customWebView.evaluateJavascript("javascript:setPositionStatus(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.5
                @Override // android.webkit.ValueCallback
                public void onReceiveValue(String str) {
                    LogUtil.w("setPositionStatus Receive:" + str);
                }
            });
        }
    }

    private void refreshSpeedModeUI(int i) {
        if (i <= 0 || i > this.mModes.size()) {
            this.layout_speed_mode.setVisibility(8);
            this.img_speed_mode.setImageDrawable(null);
        } else {
            this.layout_speed_mode.setVisibility(0);
            this.img_speed_mode.setImageDrawable(null);
            Glide.with((FragmentActivity) this.baseActivity).load(this.mModes.get(i - 1)).into(this.img_speed_mode);
        }
        if (getParentFragment() instanceof PlaySMSFragment2) {
            ((PlaySMSFragment2) getParentFragment()).refreshSpeedModeUI(i);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedUI(int i, int i2) {
        LogUtil.e("refreshSpeedUI:" + i);
        if (i2 == 1) {
            this.progress_speed.setProgress(i);
        }
        this.web_device.evaluateJavascript("javascript:setSpeed(" + i + ", " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.7
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setSpeed Receive:" + str);
            }
        });
        refreshSpeedModeUI(0);
    }

    private void refreshVibrateModeUI(int i) {
        if (i <= 0 || i > this.mVibrateModes.size()) {
            this.layout_vibrate_mode.setVisibility(8);
            this.img_vibrate_mode.setImageDrawable(null);
        } else {
            this.layout_vibrate_mode.setVisibility(0);
            this.img_vibrate_mode.setImageDrawable(null);
            Glide.with((FragmentActivity) this.baseActivity).load(this.mVibrateModes.get(i - 1)).into(this.img_vibrate_mode);
        }
        if (getParentFragment() instanceof PlaySMSFragment2) {
            ((PlaySMSFragment2) getParentFragment()).refreshVibrateModeUI(i);
        }
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_SVG_DOWNLOAD);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.web_device.setOnLongClickListener(new View.OnLongClickListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.2
            @Override // android.view.View.OnLongClickListener
            public boolean onLongClick(View view) {
                return true;
            }
        });
        this.playView.setOnLineChangeListener(new PlayView_Four.OnLineChangeListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.3
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
            public void onLineChange(int i, JsonObject jsonObject) {
                LogUtil.e("onLineChange speed:" + i);
                if (i < 0) {
                    i = 0;
                } else if (i > 100) {
                    i = 100;
                }
                if (i != PlaySMYouFragment.this.progress_speed.getProgress() || i == 0) {
                    SocketManager.getInstance().setSpeed(i, jsonObject);
                    PlaySMYouFragment.this.refreshColorStyleUI(R.color.app_style_color10, 1);
                    PlaySMYouFragment.this.refreshColorStyleUI(R.color.app_style_color, 0);
                    PlaySMYouFragment.this.refreshSpeedUI(i, 1);
                    PlaySMYouFragment.this.getBinderProduct().setSpeed(i);
                    PlaySMYouFragment.this.getBinderProduct().setMode(0);
                }
            }
        });
        this.img_reversal.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMYouFragment.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    LogUtil.e("ACTION_DOWN");
                    AnimUtil.rotate(view, 1);
                    view.setRotationY(180.0f);
                    SocketManager.getInstance().setReversal(1);
                } else if (action == 1) {
                    LogUtil.e("ACTION_UP");
                    AnimUtil.rotate(view, 0);
                    view.setRotationY(0.0f);
                    SocketManager.getInstance().setReversal(0);
                }
                return true;
            }
        });
    }

    private void unregisterReceiver() {
        try {
            this.baseActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
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
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sm_you, viewGroup, false);
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
            SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel == null || socketSpeedModel.getOnlyRead() != 1) {
                return;
            }
            String action = socketSpeedModel.getAction();
            if ("setSpeed".equals(action)) {
                if (socketSpeedModel.isMotor1()) {
                    refreshSpeedUI(socketSpeedModel.getSpeed(), 0);
                    return;
                } else {
                    if (socketSpeedModel.isMotor2()) {
                        refreshVibrateModeUI(0);
                        return;
                    }
                    return;
                }
            }
            if (!"setMode".equals(action)) {
                "setReversal".equals(action);
            } else if (socketSpeedModel.isMotor1Mode()) {
                refreshSpeedModeUI(socketSpeedModel.getMode());
            } else if (socketSpeedModel.isMotor2Mode()) {
                refreshVibrateModeUI(socketSpeedModel.getMode());
            }
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
    }
}
