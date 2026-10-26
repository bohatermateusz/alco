package com.hismith.hismithapp.fragment.webfragment;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
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
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.ConstraintSet;
import androidx.fragment.app.FragmentActivity;
import com.bumptech.glide.Glide;
import com.google.gson.Gson;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.SocketPlayFragment;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.ColorUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HudongModeLocationUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CustomWebView;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMSMyFragment extends SocketPlayFragment {
    private BaseActivity baseActivity;
    private float endDownX;
    private ImageView img_auto;
    private ImageView img_speed_mode;
    private ConstraintLayout layout_position_box;
    private ConstraintLayout layout_position_max;
    private ConstraintLayout layout_position_min;
    private ConstraintLayout layout_root;
    private ConstraintLayout layout_speed_mode;
    private Gson mGson;
    private List<String> mModes;
    private int mSafeMax;
    private ProgressBar progress_position;
    private RecorderVolumeManager recorderVolumeManager;
    private float startDownX;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private CustomWebView web_device;
    private int mPositionStatus = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int intValue = ((Integer) message.obj).intValue();
            int i = message.what;
            if (i == 1) {
                PlaySMSMyFragment.this.refreshStrokeType(intValue != 161 ? 0 : 1);
            } else if (i != 2) {
                if (i == 3) {
                    PlaySMSMyFragment.this.refreshSpeedUI(intValue, 0);
                } else if (i == 4) {
                    PlaySMSMyFragment.this.refreshSpeedModeUI(intValue);
                } else if (i == 9) {
                    PlaySMSMyFragment.this.refreshSmoothUI(intValue);
                } else if (i == 10) {
                    PlaySMSMyFragment.this.refreshPositionUI(intValue);
                }
            } else if (intValue == 0) {
                PlaySMSMyFragment.this.refreshStrokeType(0);
                PlaySMSMyFragment.this.refreshSpeedUI(0, 0);
                PlaySMSMyFragment.this.refreshPositionUI(0);
            }
            return false;
        }
    });
    private int mPositionStart = 0;
    private View.OnTouchListener minOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.14
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlaySMSMyFragment.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch right:" + view.getRight());
            } else if (action == 2) {
                float x = motionEvent.getX() - PlaySMSMyFragment.this.startDownX;
                int width = PlaySMSMyFragment.this.progress_position.getWidth();
                int right = (int) ((PlaySMSMyFragment.this.progress_position.getRight() + (PlaySMSMyFragment.this.layout_position_min.getWidth() / 2)) - (PlaySMSMyFragment.this.layout_position_min.getRight() + x));
                LogUtil.e("onTouch moveX:" + x + ", right():" + right);
                int i = 0;
                if (right < 0) {
                    right = 0;
                } else if (right > width) {
                    right = width;
                }
                double d = width;
                int i2 = (int) ((right * 100.0d) / d);
                if (i2 >= PlaySMSMyFragment.this.mPositionEnd - Constans.start_stop_position_diff) {
                    int i3 = PlaySMSMyFragment.this.mPositionEnd - Constans.start_stop_position_diff;
                    i2 = i3 < 0 ? 0 : i3;
                    right = (int) ((d / 100.0d) * i2);
                }
                if (i2 > 100) {
                    i = 100;
                } else if (i2 >= 0) {
                    i = i2;
                }
                if (PlaySMSMyFragment.this.mPositionStart != i) {
                    LogUtil.e("开始位置：" + PlaySMSMyFragment.this.mPositionStart + ", right:" + right);
                    PlaySMSMyFragment.this.refreshStartBitUI(i);
                    SocketManager.getInstance().setStroke(PlaySMSMyFragment.this.mPositionStart, PlaySMSMyFragment.this.mPositionEnd, 1);
                }
            }
            return true;
        }
    };
    private int mPositionEnd = 100;
    private View.OnTouchListener maxOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.15
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlaySMSMyFragment.this.endDownX = motionEvent.getX();
            } else if (action == 2) {
                float x = motionEvent.getX() - PlaySMSMyFragment.this.endDownX;
                int width = PlaySMSMyFragment.this.progress_position.getWidth();
                int left = (int) ((PlaySMSMyFragment.this.layout_position_max.getLeft() + x) - (PlaySMSMyFragment.this.progress_position.getLeft() - (PlaySMSMyFragment.this.layout_position_max.getWidth() / 2)));
                LogUtil.e("onTouch moveX:" + x + ", left():" + left);
                int i = 0;
                if (left < 0) {
                    left = 0;
                } else if (left > width) {
                    left = width;
                }
                int width2 = (int) (100.0d - ((left * 100.0d) / PlaySMSMyFragment.this.progress_position.getWidth()));
                if (width2 <= PlaySMSMyFragment.this.mPositionStart + Constans.start_stop_position_diff) {
                    int i2 = PlaySMSMyFragment.this.mPositionStart + Constans.start_stop_position_diff;
                    width2 = i2 > 100 ? 100 : i2;
                    double d = width;
                    left = (int) (d - ((d / 100.0d) * width2));
                }
                if (width2 > 100) {
                    i = 100;
                } else if (width2 >= 0) {
                    i = width2;
                }
                if (PlaySMSMyFragment.this.mPositionEnd != i) {
                    LogUtil.e("结束位置：" + PlaySMSMyFragment.this.mPositionEnd + ", left:" + left);
                    PlaySMSMyFragment.this.refreshEndBitUI(i);
                    SocketManager.getInstance().setStroke(PlaySMSMyFragment.this.mPositionStart, PlaySMSMyFragment.this.mPositionEnd, 1);
                }
            }
            return true;
        }
    };

    public PlaySMSMyFragment(int i) {
        this.mSafeMax = 100;
        this.mSafeMax = i;
    }

    private int getStartPositionSafeMax() {
        BaseActivity baseActivity = this.baseActivity;
        return baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getStartPositionSafeMax() : baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getStartPositionSafeMax() : BleManager.getInstance(baseActivity).getmConnBleDevice().getStartPosition();
    }

    private int getStopPositionSafeMax() {
        BaseActivity baseActivity = this.baseActivity;
        return baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getStopPositionSafeMax() : baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getStopPositionSafeMax() : BleManager.getInstance(baseActivity).getmConnBleDevice().getStopPosition();
    }

    private void init() {
        this.mGson = new Gson();
        this.mModes = new ArrayList();
        initProductData();
        initWebView();
        this.recorderVolumeManager = new RecorderVolumeManager();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initMyBleDataUI() {
        LogUtil.e("initMyBleDataUI:" + BleManager.getInstance(this.baseActivity).getmConnBleDevice());
        if (BleManager.getInstance(this.baseActivity).getmConnBleDevice() != null) {
            Message message = new Message();
            message.what = 1;
            message.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getRunMode() == 1 ? 161 : 160);
            this.mHandler.sendMessage(message);
            Message message2 = new Message();
            message2.what = 3;
            message2.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed());
            this.mHandler.sendMessage(message2);
            Message message3 = new Message();
            message3.what = 4;
            message3.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMode());
            this.mHandler.sendMessage(message3);
            refreshStartBitUI(getStartPositionSafeMax());
            refreshEndBitUI(getStopPositionSafeMax());
            Message message4 = new Message();
            message4.what = 9;
            message4.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSmoothness());
            this.mHandler.sendMessage(message4);
            Message message5 = new Message();
            message5.what = 10;
            message5.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getPosition());
            this.mHandler.sendMessage(message5);
        }
    }

    private void initView(View view) {
        this.layout_root = (ConstraintLayout) view.findViewById(R.id.layout_root);
        this.web_device = (CustomWebView) view.findViewById(R.id.web_device);
        this.layout_speed_mode = (ConstraintLayout) view.findViewById(R.id.layout_speed_mode);
        this.img_speed_mode = (ImageView) view.findViewById(R.id.img_speed_mode);
        this.img_auto = (ImageView) view.findViewById(R.id.img_auto);
        this.layout_position_box = (ConstraintLayout) view.findViewById(R.id.layout_position_box);
        this.layout_position_min = (ConstraintLayout) view.findViewById(R.id.layout_position_min);
        this.layout_position_max = (ConstraintLayout) view.findViewById(R.id.layout_position_max);
        this.txt_position_min = (TextView) view.findViewById(R.id.txt_position_min);
        this.txt_position_max = (TextView) view.findViewById(R.id.txt_position_max);
        this.progress_position = (ProgressBar) view.findViewById(R.id.progress_position);
    }

    private void initWebView() {
        String str;
        this.web_device.getSettings().setCacheMode(1);
        this.web_device.setBackgroundColor(0);
        this.web_device.setInitialScale(100);
        WebSettings settings = this.web_device.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setJavaScriptCanOpenWindowsAutomatically(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        if (MyApplication.getApplication().getmProductModel() != null) {
            String svgName = MyApplication.getApplication().getmProductModel().getSvgName();
            HudongModeLocationUtil.initModePosition(svgName, this.layout_root, R.id.web_device, R.id.layout_speed_mode, -1, 0);
            if (MyApplication.getApplication().getmProductModel().getSvgLeftRight() == 1) {
                svgName = svgName + "_right";
            }
            LogUtil.e("SVG svgName:" + svgName);
            if (CacheUtil.playSVGExists(this.baseActivity, svgName)) {
                LogUtil.e("SVG svgName存在");
                str = "file:///" + new File(this.baseActivity.getFilesDir() + "/playSVG", svgName + ".svg").getPath();
            } else {
                LogUtil.e("SVG svgName不存在");
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
            }
        } else {
            str = "file:///android_asset/01device_right.svg";
        }
        if (TextUtils.isEmpty(str)) {
            LogUtil.e("webView uri:NULL");
        } else {
            LogUtil.e("webView uri:" + str);
            this.web_device.loadUrl(str);
        }
        this.web_device.setWebViewClient(new WebViewClient() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.2
            @Override // android.webkit.WebViewClient
            public void onPageFinished(WebView webView, String str2) {
                super.onPageFinished(webView, str2);
                int i = PlaySMSMyFragment.this.mPositionStatus;
                PlaySMSMyFragment playSMSMyFragment = PlaySMSMyFragment.this;
                if (i == 1) {
                    playSMSMyFragment.layout_position_box.setVisibility(0);
                    PlaySMSMyFragment.this.img_auto.setVisibility(0);
                } else {
                    playSMSMyFragment.layout_position_box.setVisibility(8);
                    PlaySMSMyFragment.this.img_auto.setVisibility(8);
                }
                PlaySMSMyFragment.this.refreshColorStyleUI(R.color.app_style_color10, 1);
                PlaySMSMyFragment.this.refreshColorStyleUI(R.color.app_style_color, 0);
                PlaySMSMyFragment playSMSMyFragment2 = PlaySMSMyFragment.this;
                playSMSMyFragment2.refreshPositionStatus(playSMSMyFragment2.mPositionStatus);
                PlaySMSMyFragment.this.initMyBleDataUI();
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshColorStyleUI(int i, int i2) {
        this.web_device.evaluateJavascript("javascript:setColorStyle('" + ColorUtil.toHexEncoding(this.baseActivity, i) + "', " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.7
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setColorStyle Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        LogUtil.e("refreshEndBitUI:" + i);
        this.mPositionEnd = i;
        setStopPositionSafeMax(i);
        this.txt_position_max.setText(this.mPositionEnd + "");
        double width = this.progress_position.getWidth();
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_max.getLayoutParams();
        layoutParams.setMargins((int) (width - ((width / 100.0d) * i)), 0, 0, 0);
        this.layout_position_max.setLayoutParams(layoutParams);
        this.web_device.evaluateJavascript("javascript:setPositionEnd(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.11
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
            customWebView.evaluateJavascript("javascript:setPositionStatus(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.8
                @Override // android.webkit.ValueCallback
                public void onReceiveValue(String str) {
                    LogUtil.w("setPositionStatus Receive:" + str);
                }
            });
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionUI(int i) {
        this.web_device.evaluateJavascript("javascript:setPosition('" + LanguageUtil.getLanguageByKey(this.baseActivity, "position") + "' , " + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.13
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setPosition Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSmoothUI(int i) {
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
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedUI(int i, int i2) {
        LogUtil.e("refreshSpeedUI:" + i);
        this.web_device.evaluateJavascript("javascript:setSpeed(" + i + ", " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.9
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
        this.mPositionStart = i;
        setStartPositionSafeMax(i);
        this.txt_position_min.setText(this.mPositionStart + "");
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_min.getLayoutParams();
        layoutParams.setMargins(0, 0, (int) ((this.progress_position.getWidth() / 100.0d) * i), 0);
        this.layout_position_min.setLayoutParams(layoutParams);
        this.web_device.evaluateJavascript("javascript:setPositionStart(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.12
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setPositionStart Receive:" + str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStrokeType(int i) {
        ImageView imageView = this.img_auto;
        if (i == 1) {
            imageView.setImageResource(R.mipmap.ic_auto_switch_off);
        } else {
            imageView.setImageResource(R.mipmap.ic_auto_switch_on);
        }
        LogUtil.e("refreshStrokeType:" + i);
        this.web_device.evaluateJavascript("javascript:setStrokeType(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.10
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setStrokeType Receive:" + str);
            }
        });
    }

    private void setListener() {
        this.web_device.setOnLongClickListener(new View.OnLongClickListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.3
            @Override // android.view.View.OnLongClickListener
            public boolean onLongClick(View view) {
                return true;
            }
        });
        this.img_auto.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (BleManager.getInstance(PlaySMSMyFragment.this.baseActivity).getmConnBleDevice() == null || BleManager.getInstance(PlaySMSMyFragment.this.baseActivity).getmConnBleDevice().getRunMode() != 0) {
                    Message message = new Message();
                    message.what = 1;
                    message.obj = 160;
                    PlaySMSMyFragment.this.mHandler.sendMessage(message);
                    BleManager.getInstance(PlaySMSMyFragment.this.baseActivity).writeRunAuto();
                    return;
                }
                Message message2 = new Message();
                message2.what = 1;
                message2.obj = 161;
                PlaySMSMyFragment.this.mHandler.sendMessage(message2);
                BleManager.getInstance(PlaySMSMyFragment.this.baseActivity).writeRunManual();
            }
        });
        this.layout_position_min.setOnTouchListener(this.minOnTouch);
        this.layout_position_max.setOnTouchListener(this.maxOnTouch);
    }

    private void setStartPositionSafeMax(int i) {
        BaseActivity baseActivity = this.baseActivity;
        if (baseActivity instanceof ControlAcceptActivity2) {
            ((ControlAcceptActivity2) baseActivity).setStartPositionSafeMax(i);
        } else if (baseActivity instanceof ControlSponsorActivity2) {
            ((ControlSponsorActivity2) baseActivity).setStartPositionSafeMax(i);
        }
    }

    private void setStopPositionSafeMax(int i) {
        BaseActivity baseActivity = this.baseActivity;
        if (baseActivity instanceof ControlAcceptActivity2) {
            ((ControlAcceptActivity2) baseActivity).setStopPositionSafeMax(i);
        } else if (baseActivity instanceof ControlSponsorActivity2) {
            ((ControlSponsorActivity2) baseActivity).setStopPositionSafeMax(i);
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initBleCCData(String str, String str2, String str3) {
        try {
            LogUtil.e("data0:" + str2 + ", data1:" + str3);
            Message message = new Message();
            message.what = Integer.valueOf(str2, 16).intValue();
            message.obj = Integer.valueOf(str3, 16);
            this.mHandler.sendMessage(message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initProductData() {
        if (MyApplication.getApplication().getmProductModel() == null) {
            return;
        }
        this.mModes.clear();
        if (ProductUtil.hasMode(0)) {
            for (ProductModel.Mode mode : ProductUtil.getProductMode(0)) {
                this.mModes.add(mode.getmIcon());
            }
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment, com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        BaseActivity baseActivity = (BaseActivity) getActivity();
        this.baseActivity = baseActivity;
        if (baseActivity instanceof ControlAcceptActivity2) {
            LogUtil.e("PlaySMSMyFragment ControlAcceptActivity2");
        } else if (baseActivity instanceof ControlSponsorActivity2) {
            LogUtil.e("PlaySMSMyFragment ControlSponsorActivity2");
        } else {
            LogUtil.e("PlaySMSMyFragment else");
        }
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sms_my, viewGroup, false);
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

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onSetSafeMax(int i) {
        CustomWebView customWebView = this.web_device;
        if (customWebView != null) {
            customWebView.evaluateJavascript("javascript:setMaxSpeed(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.5
                @Override // android.webkit.ValueCallback
                public void onReceiveValue(String str) {
                    LogUtil.w("setMaxSpeed Receive:" + str);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onSetVibrateSafeMax(int i) {
        CustomWebView customWebView = this.web_device;
        if (customWebView != null) {
            customWebView.evaluateJavascript("javascript:setMaxVibrate(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMSMyFragment.6
                @Override // android.webkit.ValueCallback
                public void onReceiveValue(String str) {
                    LogUtil.w("setMaxVibrate Receive:" + str);
                }
            });
        }
    }
}
