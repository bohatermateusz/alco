package com.hismith.hismithapp.fragment.webfragment;

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
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ImageView;
import android.widget.ProgressBar;
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
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.ColorUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HudongModeLocationUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMSYouFragment extends SocketPlayFragment {
    private BaseActivity baseActivity;
    private ImageView img_position;
    private ImageView img_speed_mode;
    private ConstraintLayout layout_root;
    private ConstraintLayout layout_speed_mode;
    private ConstraintLayout layout_touch_position;
    private Gson mGson;
    private List<String> mModes;
    private int mPosition;
    private SocketProductModel mProductModel;
    private PlayView_Four playView;
    private ProgressBar progress_position;
    private VerticalProgressBar progress_speed;
    private CustomWebView web_device;
    private int mPositionStatus = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int intValue = ((Integer) message.obj).intValue();
            int i = message.what;
            if (i == 1) {
                PlaySMSYouFragment.this.refreshStrokeType(intValue);
            } else if (i == 3) {
                PlaySMSYouFragment.this.refreshSpeedUI(intValue, 0);
            } else if (i != 4) {
                switch (i) {
                    case 7:
                        PlaySMSYouFragment.this.refreshStartBitUI(intValue);
                        break;
                    case 8:
                        PlaySMSYouFragment.this.refreshEndBitUI(intValue);
                        break;
                    case 9:
                        PlaySMSYouFragment.this.refreshSmoothUI(intValue);
                        break;
                    case 10:
                        PlaySMSYouFragment.this.refreshPositionUI(intValue);
                        break;
                }
            } else {
                PlaySMSYouFragment.this.refreshSpeedModeUI(intValue);
            }
            return false;
        }
    });
    private View.OnTouchListener positionOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.13
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action != 0 && action != 2) {
                return true;
            }
            PlaySMSYouFragment playSMSYouFragment = PlaySMSYouFragment.this;
            int changeUIPosition = playSMSYouFragment.changeUIPosition(playSMSYouFragment.layout_touch_position, PlaySMSYouFragment.this.img_position, motionEvent.getX(), motionEvent.getY());
            if (changeUIPosition > 100) {
                changeUIPosition = 100;
            } else if (changeUIPosition < 0) {
                changeUIPosition = 0;
            }
            SocketManager.getInstance().setStrokeNow(changeUIPosition);
            LogUtil.e("position:" + changeUIPosition + ", ");
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.14
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (Constans.RECEIVER_SVG_DOWNLOAD.equals(intent.getAction())) {
                PlaySMSYouFragment.this.initWebView();
            }
        }
    };

    public class JsInterface {
        public JsInterface() {
        }

        @JavascriptInterface
        public void sendMsg(String str) {
            LogUtil.e("SVG sendMsg 滑块位置变化: " + str);
            SocketManager.getInstance().send(str);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public int changeUIPosition(View view, View view2, float f, float f2) {
        int width = (int) (f - (view2.getWidth() / 2.0f));
        int height = (int) (f2 - (view2.getHeight() / 2.0f));
        int width2 = view2.getWidth() + width;
        int height2 = view2.getHeight() + height;
        if (width < 0) {
            width2 = view2.getWidth();
            width = 0;
        }
        if (width2 > view.getWidth()) {
            width2 = view.getWidth();
            width = width2 - view2.getWidth();
        }
        if (height < 0) {
            height2 = view2.getHeight();
            height = 0;
        }
        if (height2 > view.getHeight()) {
            height2 = view.getHeight();
            height = height2 - view2.getHeight();
        }
        view2.layout(width, height, width2, height2);
        return 100 - ((int) (((height * 1.0f) / (this.layout_touch_position.getHeight() - view2.getHeight())) * 100.0f));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public SocketProductModel getBinderProduct() {
        BaseActivity baseActivity = this.baseActivity;
        return baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getBinder().getSocketProductModel() : baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getBinder().getSocketProductModel() : new SocketProductModel();
    }

    private void init() {
        this.mGson = new Gson();
        this.mModes = new ArrayList();
        Bundle arguments = getArguments();
        if (arguments != null) {
            this.mProductModel = (SocketProductModel) arguments.getSerializable("ProductModel");
        }
        SocketProductModel socketProductModel = this.mProductModel;
        if (socketProductModel != null) {
            initProduct(socketProductModel);
        }
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleDataUI(SocketProductModel socketProductModel) {
        if (socketProductModel != null) {
            Message message = new Message();
            message.what = 1;
            message.obj = Integer.valueOf(socketProductModel.getStrokeType());
            this.mHandler.sendMessage(message);
            Message message2 = new Message();
            message2.what = 3;
            message2.obj = Integer.valueOf(socketProductModel.getSpeed());
            this.mHandler.sendMessage(message2);
            Message message3 = new Message();
            message3.what = 4;
            message3.obj = Integer.valueOf(socketProductModel.getMode());
            this.mHandler.sendMessage(message3);
            Message message4 = new Message();
            message4.what = 7;
            message4.obj = Integer.valueOf(socketProductModel.getStrokeMin());
            this.mHandler.sendMessage(message4);
            Message message5 = new Message();
            message5.what = 8;
            message5.obj = Integer.valueOf(socketProductModel.getStrokeMax());
            this.mHandler.sendMessage(message5);
            Message message6 = new Message();
            message6.what = 9;
            message6.obj = Integer.valueOf(socketProductModel.getSmooth());
            this.mHandler.sendMessage(message6);
            Message message7 = new Message();
            message7.what = 10;
            message7.obj = Integer.valueOf(socketProductModel.getStrokeNow());
            this.mHandler.sendMessage(message7);
        }
    }

    private void initProduct(SocketProductModel socketProductModel) {
        if (socketProductModel != null) {
            PlayView_Four playView_Four = this.playView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(socketProductModel.getSlide());
            }
            if (socketProductModel.getModesUrl() != null) {
                this.mModes.addAll(ArrayUtils.toArrayList(socketProductModel.getModesUrl().getModes()));
            }
        }
    }

    private void initView(View view) {
        this.layout_root = (ConstraintLayout) view.findViewById(R.id.layout_root);
        this.web_device = (CustomWebView) view.findViewById(R.id.web_device);
        this.progress_position = (ProgressBar) view.findViewById(R.id.progress_position);
        this.progress_speed = (VerticalProgressBar) view.findViewById(R.id.progress_speed);
        this.playView = (PlayView_Four) view.findViewById(R.id.playView);
        this.layout_touch_position = (ConstraintLayout) view.findViewById(R.id.layout_touch_position);
        this.img_position = (ImageView) view.findViewById(R.id.img_position);
        this.layout_speed_mode = (ConstraintLayout) view.findViewById(R.id.layout_speed_mode);
        this.img_speed_mode = (ImageView) view.findViewById(R.id.img_speed_mode);
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
                this.mPositionStatus = 0;
            } else if (svgName.contains("08device")) {
                ConstraintSet constraintSet2 = new ConstraintSet();
                constraintSet2.clone(this.layout_root);
                constraintSet2.constrainPercentWidth(R.id.web_device, 0.34f);
                constraintSet2.setDimensionRatio(R.id.web_device, "H,1:0.788");
                constraintSet2.applyTo(this.layout_root);
                this.mPositionStatus = 0;
            } else {
                ConstraintSet constraintSet3 = new ConstraintSet();
                constraintSet3.clone(this.layout_root);
                constraintSet3.constrainPercentWidth(R.id.web_device, 0.5f);
                constraintSet3.setDimensionRatio(R.id.web_device, "H,1:0.383");
                constraintSet3.applyTo(this.layout_root);
                this.mPositionStatus = 1;
            }
        } else {
            str = "file:///android_asset/01device_left.svg";
        }
        this.web_device.addJavascriptInterface(new JsInterface(), "appWindow");
        if (TextUtils.isEmpty(str)) {
            LogUtil.e("webView uri:NULL");
        } else {
            LogUtil.e("webView uri:" + str);
            this.web_device.loadUrl(str);
        }
        this.web_device.setWebViewClient(new WebViewClient() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.2
            @Override // android.webkit.WebViewClient
            public void onPageFinished(WebView webView, String str2) {
                super.onPageFinished(webView, str2);
                PlaySMSYouFragment.this.refreshColorStyleUI(R.color.app_style_color10, 1);
                PlaySMSYouFragment.this.refreshColorStyleUI(R.color.app_style_color, 0);
                PlaySMSYouFragment playSMSYouFragment = PlaySMSYouFragment.this;
                playSMSYouFragment.refreshPositionStatus(playSMSYouFragment.mPositionStatus);
                PlaySMSYouFragment playSMSYouFragment2 = PlaySMSYouFragment.this;
                playSMSYouFragment2.initBleDataUI(playSMSYouFragment2.mProductModel);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshColorStyleUI(int i, int i2) {
        this.web_device.evaluateJavascript("javascript:setColorStyle('" + ColorUtil.toHexEncoding(this.baseActivity, i) + "', " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.5
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setColorStyle Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        LogUtil.e("refreshEndBitUI:" + i);
        this.web_device.evaluateJavascript("javascript:setPositionEnd(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.9
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setPositionEnd Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionStatus(int i) {
        CustomWebView customWebView = this.web_device;
        if (customWebView != null) {
            customWebView.evaluateJavascript("javascript:setPositionStatus(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.6
                @Override // android.webkit.ValueCallback
                public void onReceiveValue(String str) {
                    LogUtil.w("setPositionStatus Receive:" + str);
                }
            });
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionUI(int i) {
        this.mPosition = i;
        this.web_device.evaluateJavascript("javascript:setPosition('" + LanguageUtil.getLanguageByKey(this.baseActivity, "position") + "' , " + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.12
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setPosition Receive:" + str);
            }
        });
        this.progress_position.setProgress(i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSmoothUI(int i) {
        this.web_device.evaluateJavascript("javascript:setSmoothness(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.11
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setSmoothness Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedModeUI(int i) {
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
        this.web_device.evaluateJavascript("javascript:setSpeed(" + i + ", " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.7
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setSpeed Receive:" + str);
            }
        });
        refreshSpeedModeUI(0);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartBitUI(int i) {
        LogUtil.e("refreshStartBitUI:" + i);
        this.web_device.evaluateJavascript("javascript:setPositionStart(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.10
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setPositionStart Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStrokeType(int i) {
        PlayView_Four playView_Four = this.playView;
        if (i == 1) {
            playView_Four.setTouch(false);
            this.layout_touch_position.setVisibility(0);
        } else {
            playView_Four.setTouch(true);
            this.layout_touch_position.setVisibility(8);
        }
        LogUtil.e("refreshStrokeType:" + i);
        this.web_device.evaluateJavascript("javascript:setStrokeType(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.8
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setStrokeType Receive:" + str);
            }
        });
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_SVG_DOWNLOAD);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.web_device.setOnLongClickListener(new View.OnLongClickListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.3
            @Override // android.view.View.OnLongClickListener
            public boolean onLongClick(View view) {
                return true;
            }
        });
        this.layout_touch_position.setOnTouchListener(this.positionOnTouch);
        this.playView.setOnLineChangeListener(new PlayView_Four.OnLineChangeListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSYouFragment.4
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
            public void onLineChange(int i, JsonObject jsonObject) {
                LogUtil.e("onLineChange speed:" + i);
                if (i < 0) {
                    i = 0;
                } else if (i > 100) {
                    i = 100;
                }
                if (i != PlaySMSYouFragment.this.progress_speed.getProgress() || i == 0) {
                    SocketManager.getInstance().setSpeed(i, jsonObject);
                    PlaySMSYouFragment.this.progress_speed.setProgress(i);
                    PlaySMSYouFragment.this.refreshSpeedUI(i, 1);
                    PlaySMSYouFragment.this.getBinderProduct().setSpeed(i);
                    PlaySMSYouFragment.this.getBinderProduct().setMode(0);
                }
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
        initWebView();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sms_you, viewGroup, false);
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
            SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel == null || socketSpeedModel.getOnlyRead() != 1) {
                return;
            }
            String action = socketSpeedModel.getAction();
            if ("setStrokeNow".equals(action)) {
                refreshPositionUI(socketSpeedModel.getStrokeNow());
                return;
            }
            if ("setStrokeType".equals(action)) {
                refreshStrokeType(socketSpeedModel.getStrokeType());
                return;
            }
            if ("setStroke".equals(action)) {
                refreshEndBitUI(socketSpeedModel.getStrokeMax());
                refreshStartBitUI(socketSpeedModel.getStrokeMin());
                return;
            }
            if ("setSpeed".equals(action)) {
                if (socketSpeedModel.isMotor1()) {
                    refreshSpeedUI(socketSpeedModel.getSpeed(), 0);
                    return;
                } else {
                    socketSpeedModel.isMotor2();
                    return;
                }
            }
            if ("setMode".equals(action)) {
                if (socketSpeedModel.isMotor1Mode()) {
                    refreshSpeedModeUI(socketSpeedModel.getMode());
                } else {
                    socketSpeedModel.isMotor2Mode();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
