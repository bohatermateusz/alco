package com.hismith.hismithapp.fragment.webfragment;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.LayoutInflater;
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
import com.google.gson.Gson;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.fragment.SocketPlayFragment;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.ColorUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HudongModeLocationUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CustomWebView;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMVMyFragment extends SocketPlayFragment {
    private BaseActivity baseActivity;
    private ImageView img_speed_mode;
    private ImageView img_vibrate_mode;
    private ConstraintLayout layout_root;
    private ConstraintLayout layout_speed_mode;
    private ConstraintLayout layout_vibrate_mode;
    private Gson mGson;
    private List<ProductModel.Mode> mModes;
    private int mSafeMax;
    private int mVibrateSafeMax;
    private List<ProductModel.Mode> mVibrationModes;
    private CustomWebView web_device;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_VIBRATION_MODE = 5;
    private final int WHAT_MUSIC = 6;
    private final int WHAT_VIBRATE = 7;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                PlaySMVMyFragment.this.refreshSpeedUI(((Integer) message.obj).intValue(), 0);
            } else if (i == 2) {
                PlaySMVMyFragment.this.refreshSpeedModeUI(((Integer) message.obj).intValue());
            } else if (i == 5) {
                PlaySMVMyFragment.this.refreshVibrateModeUI(((Integer) message.obj).intValue());
            } else if (i == 7) {
                PlaySMVMyFragment.this.refreshVibrateUI(((Integer) message.obj).intValue(), 0);
            }
            return false;
        }
    });

    public PlaySMVMyFragment(int i, int i2) {
        this.mSafeMax = 100;
        this.mVibrateSafeMax = 100;
        this.mSafeMax = i;
        this.mVibrateSafeMax = i2;
    }

    private void init() {
        this.mGson = new Gson();
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        initProductData();
        initWebView();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initMyBleDataUI() {
        if (BleManager.getInstance(this.baseActivity).getmConnBleDevice() != null) {
            Message message = new Message();
            message.what = 1;
            message.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed());
            this.mHandler.sendMessage(message);
            Message message2 = new Message();
            message2.what = 2;
            message2.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMode());
            this.mHandler.sendMessage(message2);
            Message message3 = new Message();
            message3.what = 7;
            message3.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibration());
            this.mHandler.sendMessage(message3);
            Message message4 = new Message();
            message4.what = 5;
            message4.obj = Integer.valueOf(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibrationMode());
            this.mHandler.sendMessage(message4);
        }
    }

    private void initView(View view) {
        this.layout_root = (ConstraintLayout) view.findViewById(R.id.layout_root);
        this.web_device = (CustomWebView) view.findViewById(R.id.web_device);
        this.layout_speed_mode = (ConstraintLayout) view.findViewById(R.id.layout_speed_mode);
        this.layout_vibrate_mode = (ConstraintLayout) view.findViewById(R.id.layout_vibrate_mode);
        this.img_speed_mode = (ImageView) view.findViewById(R.id.img_speed_mode);
        this.img_vibrate_mode = (ImageView) view.findViewById(R.id.img_vibrate_mode);
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
            HudongModeLocationUtil.initModePosition(svgName, this.layout_root, R.id.web_device, R.id.layout_speed_mode, R.id.layout_vibrate_mode, 0);
            if (MyApplication.getApplication().getmProductModel().getSvgLeftRight() == 1) {
                svgName = svgName + "_right";
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
            str = "file:///android_asset/01device_right.svg";
        }
        if (TextUtils.isEmpty(str)) {
            LogUtil.e("webView uri:NULL");
        } else {
            LogUtil.e("webView uri:" + str);
            this.web_device.loadUrl(str);
        }
        this.web_device.setWebViewClient(new WebViewClient() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.2
            @Override // android.webkit.WebViewClient
            public void onPageFinished(WebView webView, String str2) {
                super.onPageFinished(webView, str2);
                PlaySMVMyFragment.this.refreshColorStyleUI(R.color.app_style_color10, 1);
                PlaySMVMyFragment.this.refreshColorStyleUI(R.color.app_style_color, 0);
                PlaySMVMyFragment.this.initMyBleDataUI();
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshColorStyleUI(int i, final int i2) {
        this.web_device.evaluateJavascript("javascript:setColorStyle('" + ColorUtil.toHexEncoding(this.baseActivity, i) + "', " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.4
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setColorStyle Receive:" + str + ", type:" + i2);
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
            Glide.with((FragmentActivity) this.baseActivity).load(this.mModes.get(i - 1).getmIcon()).into(this.img_speed_mode);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedUI(int i, int i2) {
        this.web_device.evaluateJavascript("javascript:setSpeed(" + i + ", " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.5
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setSpeed Receive:" + str);
            }
        });
        refreshSpeedModeUI(0);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshVibrateModeUI(int i) {
        if (i <= 0 || i > this.mVibrationModes.size()) {
            this.layout_vibrate_mode.setVisibility(8);
            this.img_vibrate_mode.setImageDrawable(null);
        } else {
            this.layout_vibrate_mode.setVisibility(0);
            this.img_vibrate_mode.setImageDrawable(null);
            Glide.with((FragmentActivity) this.baseActivity).load(this.mVibrationModes.get(i - 1).getmIcon()).into(this.img_vibrate_mode);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshVibrateUI(int i, int i2) {
        this.web_device.evaluateJavascript("javascript:setVibrate(" + i + ", " + i2 + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.6
            @Override // android.webkit.ValueCallback
            public void onReceiveValue(String str) {
                LogUtil.w("setVibrate Receive:" + str);
            }
        });
        refreshVibrateModeUI(0);
    }

    private void setListener() {
        this.web_device.setOnLongClickListener(new View.OnLongClickListener() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.3
            @Override // android.view.View.OnLongClickListener
            public boolean onLongClick(View view) {
                return true;
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initBleAAData(String str, String str2) {
        int i;
        str.hashCode();
        i = 0;
        switch (str) {
            case "02":
                try {
                    Message message = new Message();
                    message.what = 1;
                    message.obj = 0;
                    this.mHandler.sendMessage(message);
                    Message message2 = new Message();
                    message2.what = 7;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    Message message3 = new Message();
                    message3.what = 6;
                    message3.obj = 0;
                    this.mHandler.sendMessage(message3);
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            case "03":
            case "04":
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message4 = new Message();
                    message4.what = 1;
                    message4.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message4);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "05":
                try {
                    int intValue2 = Integer.valueOf(str2, 16).intValue();
                    Message message5 = new Message();
                    message5.what = 2;
                    message5.obj = Integer.valueOf(intValue2);
                    this.mHandler.sendMessage(message5);
                    break;
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
            case "06":
                try {
                    if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getClimax() != 1 && Constans.DEVICE_TYPE_SM2.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
                        int intValue3 = Integer.valueOf(str2, 16).intValue();
                        Message message6 = new Message();
                        message6.what = 5;
                        message6.obj = Integer.valueOf(intValue3);
                        this.mHandler.sendMessage(message6);
                        break;
                    }
                } catch (Exception e4) {
                    e4.printStackTrace();
                    return;
                }
                break;
            case "07":
                try {
                    if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getMusic() == 1) {
                        Message message7 = new Message();
                        message7.what = 6;
                        message7.obj = Integer.valueOf(str2, 16);
                        this.mHandler.sendMessage(message7);
                        break;
                    }
                } catch (Exception e5) {
                    e5.printStackTrace();
                    return;
                }
                break;
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initBleCCData(String str, String str2, String str3) {
        int i;
        str2.hashCode();
        i = 0;
        switch (str2) {
            case "01":
                try {
                    Message message = new Message();
                    if (!"B0".equals(str3)) {
                        if ("B1".equals(str3)) {
                            message.what = 6;
                            message.obj = 1;
                            this.mHandler.sendMessage(message);
                            break;
                        }
                    } else {
                        message.what = 6;
                        message.obj = 0;
                        this.mHandler.sendMessage(message);
                        break;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;
            case "02":
                try {
                    Message message2 = new Message();
                    message2.what = 1;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    Message message3 = new Message();
                    message3.what = 7;
                    message3.obj = 0;
                    this.mHandler.sendMessage(message3);
                    Message message4 = new Message();
                    message4.what = 6;
                    message4.obj = 0;
                    this.mHandler.sendMessage(message4);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "03":
                try {
                    int intValue = Integer.valueOf(str3, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message5 = new Message();
                    message5.what = 1;
                    message5.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message5);
                    break;
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
            case "04":
                try {
                    int intValue2 = Integer.valueOf(str3, 16).intValue();
                    Message message6 = new Message();
                    message6.what = 2;
                    message6.obj = Integer.valueOf(intValue2);
                    this.mHandler.sendMessage(message6);
                    break;
                } catch (Exception e4) {
                    e4.printStackTrace();
                    return;
                }
            case "05":
                try {
                    int intValue3 = Integer.valueOf(str3, 16).intValue();
                    if (intValue3 > 100) {
                        i = 100;
                    } else if (intValue3 >= 0) {
                        i = intValue3;
                    }
                    Message message7 = new Message();
                    message7.what = 7;
                    message7.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message7);
                    break;
                } catch (Exception e5) {
                    e5.printStackTrace();
                    return;
                }
            case "06":
                try {
                    int intValue4 = Integer.valueOf(str3, 16).intValue();
                    Message message8 = new Message();
                    message8.what = 5;
                    message8.obj = Integer.valueOf(intValue4);
                    this.mHandler.sendMessage(message8);
                    break;
                } catch (Exception e6) {
                    e6.printStackTrace();
                    return;
                }
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initProductData() {
        if (MyApplication.getApplication().getmProductModel() == null) {
            return;
        }
        this.mModes.clear();
        if (ProductUtil.hasMode(0)) {
            this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
        }
        this.mVibrationModes.clear();
        if (ProductUtil.hasMode(1)) {
            this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
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
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_smv_my, viewGroup, false);
        initView(inflate);
        return inflate;
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onNotify(String str) {
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onSetSafeMax(int i) {
        CustomWebView customWebView = this.web_device;
        if (customWebView != null) {
            customWebView.evaluateJavascript("javascript:setMaxSpeed(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.7
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
            customWebView.evaluateJavascript("javascript:setMaxVibrate(" + i + ")", new ValueCallback<String>() { // from class: com.hismith.hismithapp.fragment.webfragment.PlaySMVMyFragment.8
                @Override // android.webkit.ValueCallback
                public void onReceiveValue(String str) {
                    LogUtil.w("setMaxVibrate Receive:" + str);
                }
            });
        }
    }
}
