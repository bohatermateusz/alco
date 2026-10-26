package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.animation.AnimationUtils;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.facebook.appevents.integrity.IntegrityManager;
import com.facebook.internal.NativeProtocol;
import com.hismith.hismithapp.adapter.PlayModeListAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.ModeBlockModel;
import com.hismith.hismithapp.model.ModeSettingModel;
import com.hismith.hismithapp.model.PlayModeCreateModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.surface_view.LineDataUtil;
import com.hismith.hismithapp.surface_view.LineDrawView5;
import com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener;
import com.hismith.hismithapp.surface_view.listener.OnRecodeTimeListener;
import com.hismith.hismithapp.surface_view.listener.OnSelectTimeListener;
import com.hismith.hismithapp.surface_view.listener.OnStatusChangeListener;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.ModeTimeUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.DoubleSliderView;
import com.hismith.hismithapp.view.NodeSliderView;
import com.hismith.hismithapp.view.PlaySettingImageView;
import com.hismith.hismithapp.view.TouchLayout;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayModeCreateActivity2 extends BaseActivity implements View.OnClickListener {
    private static final int WHAT_PLAY_MODE_MANUAL = 2;
    private static final int WHAT_PLAY_MODE_SMOOTH = 5;
    private static final int WHAT_PLAY_MODE_START_PIS = 3;
    private static final int WHAT_PLAY_MODE_STOP_PIS = 4;
    private String action;
    private RetrofitRequest baseRequest;
    private ImageView img_piu;
    private ImageView img_reversal;
    private LinearLayout layout_mark;
    private PlayModeListAdapter mAdapter;
    private BothWayArrowView mArrowBoth;
    private ActivityResultLauncher mBluetoothLauncher;
    private TextView mBtnSave;
    private ImageView mImgGesture;
    private ImageView mImgModeCopy;
    private ImageView mImgModeDel;
    private ImageView mImgModeList;
    private ImageView mImgModePaste;
    private ImageView mImgModePlay;
    private ImageView mImgModeSelect;
    private ImageView mImgModeStart;
    private ImageView mImgModeStop;
    private PlaySettingImageView mImgSetting;
    private ImageView mImgTouch;
    private LinearLayout mLayoutArrow;
    private LinearLayoutManager mLayoutManager;
    private LinearLayout mLayoutModeList;
    private FrameLayout mLayoutTouch;
    private LineDrawView5 mLineDrawView;
    private List<ModeBlockModel.ModeBlock> mList;
    private RecyclerView mRecyclerMode;
    private SmartRefreshLayout mRefreshLayoutMode;
    private String mRequestId;
    private SeekBar mSeekTime;
    private ImageView mTitleBack;
    private ImageView mTitleBluetooth;
    private TextView mTitleName;
    private FrameLayout mTouchBgLayout;
    private FrameLayout mTouchBgLayout1;
    private TouchLayout mTouchLayout;
    private TouchLayout mTouchLayout1;
    private TextView mTxtEndTime;
    private TextView mTxtModeList;
    private TextView mTxtStartTime;
    private DoubleSliderView slider_position;
    private NodeSliderView slider_smooth;
    private TextView txt_mode_depth;
    private TextView txt_mode_max_stroke;
    private TextView txt_mode_smooth;
    private TextView txt_mode_speed;
    private PlayModeModel mEditMode = null;
    private boolean isTouchSeek = false;
    private int lastSpeed = 0;
    private int lastVibrate = 0;
    private int lastStartPosition = 0;
    private int lastStopPosition = 0;
    private int lastSmooth = 0;
    private int lastReversal = 0;
    private int mModeType = 2;
    private final int delay = 15;
    private int delayMillis = 0;
    private final int WHAT_PIU = 6;
    private final int WHAT_WRITE_PIU = 7;
    Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            switch (message.what) {
                case 2:
                    BleManager.getInstance(PlayModeCreateActivity2.this).writeRunManual();
                    break;
                case 3:
                    if (PlayModeCreateActivity2.this.mEditMode != null && PlayModeCreateActivity2.this.mEditMode.getModeSettingsModel() != null) {
                        BleManager.getInstance(PlayModeCreateActivity2.this).writeStartPosition(PlayModeCreateActivity2.this.mEditMode.getModeSettingsModel().getStrokeMin());
                        break;
                    }
                    break;
                case 4:
                    if (PlayModeCreateActivity2.this.mEditMode != null && PlayModeCreateActivity2.this.mEditMode.getModeSettingsModel() != null) {
                        BleManager.getInstance(PlayModeCreateActivity2.this).writeStopPosition(PlayModeCreateActivity2.this.mEditMode.getModeSettingsModel().getStrokeMax());
                        break;
                    }
                    break;
                case 5:
                    if (PlayModeCreateActivity2.this.mEditMode != null && PlayModeCreateActivity2.this.mEditMode.getModeSettingsModel() != null) {
                        BleManager.getInstance(PlayModeCreateActivity2.this).writeSmooth(PlayModeCreateActivity2.this.mEditMode.getModeSettingsModel().getSmooth());
                        break;
                    }
                    break;
                case 6:
                    int intValue = ((Integer) message.obj).intValue();
                    PlayModeCreateActivity2 playModeCreateActivity2 = PlayModeCreateActivity2.this;
                    if (intValue != 1) {
                        playModeCreateActivity2.img_piu.setImageResource(R.mipmap.ic_iplay_piu);
                        break;
                    } else {
                        playModeCreateActivity2.img_piu.setImageResource(R.mipmap.ic_iplay_piu_select);
                        break;
                    }
                case 7:
                    PlayModeCreateActivity2.this.mLineDrawView.addFunData(Constans.START_TYPE_PIU, 1);
                    PlayModeCreateActivity2.this.mHandler.sendEmptyMessageDelayed(7, 100L);
                    break;
            }
            return false;
        }
    });
    private View.OnTouchListener piuOnTouchListener = new View.OnTouchListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.16
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlayModeCreateActivity2.this.mHandler.sendEmptyMessage(7);
            } else if (action == 1) {
                PlayModeCreateActivity2.this.mHandler.removeMessages(7);
            }
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.18
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals(Constans.RECEIVER_BLE_CONN_STATE)) {
                int intExtra = intent.getIntExtra("ble_state", 0);
                PlayModeCreateActivity2 playModeCreateActivity2 = PlayModeCreateActivity2.this;
                if (intExtra == 2) {
                    playModeCreateActivity2.mTitleBluetooth.setImageDrawable(ContextCompat.getDrawable(PlayModeCreateActivity2.this, R.mipmap.ic_play_bluetooth_connect));
                    return;
                } else {
                    playModeCreateActivity2.mTitleBluetooth.setImageDrawable(ContextCompat.getDrawable(PlayModeCreateActivity2.this, R.mipmap.ic_play_bluetooth_disconnect));
                    return;
                }
            }
            if (action.equals(Constans.RECEIVER_PRODUCT_INFO)) {
                PlayModeCreateActivity2.this.initProductData();
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                String stringExtra4 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    PlayModeCreateActivity2.this.initBleCCData(stringExtra4, stringExtra2, stringExtra3);
                }
            }
        }
    };

    public class OnViewGlobalLayoutListener implements ViewTreeObserver.OnGlobalLayoutListener {
        private int maxHeight;
        private View view;

        public OnViewGlobalLayoutListener(View view, int i) {
            this.view = view;
            this.maxHeight = i;
        }

        @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
        public void onGlobalLayout() {
            if (this.view.getHeight() > this.maxHeight) {
                this.view.getLayoutParams().height = this.maxHeight;
            }
        }
    }

    private boolean bluetoothIsEnabled() {
        if (Build.VERSION.SDK_INT >= 31) {
            return true;
        }
        if (BleManager.getInstance(this).getBluetoothAdapter() == null) {
            return false;
        }
        if (BleManager.getInstance(this).getBluetoothAdapter().isEnabled()) {
            return true;
        }
        Intent intent = new Intent("android.bluetooth.adapter.action.REQUEST_ENABLE");
        ActivityResultLauncher activityResultLauncher = this.mBluetoothLauncher;
        if (activityResultLauncher == null) {
            return false;
        }
        activityResultLauncher.launch(intent);
        return false;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public String getModeData() {
        String buildToServerData = LineDataUtil.buildToServerData(this.mLineDrawView.getStrData(), this.mLineDrawView.getIntervalTime());
        String buildToServerData2 = this.mLineDrawView.isVibrateLine() ? LineDataUtil.buildToServerData(this.mLineDrawView.getStrVibrateData(), this.mLineDrawView.getIntervalTime()) : "";
        return (ProductUtil.isSMSType() && this.mModeType == 2) ? LineDataUtil.buildToServerStructure(2, this.mLineDrawView.isStartPositionLine() ? LineDataUtil.buildToServerData(this.mLineDrawView.getStrStartPositionData(), this.mLineDrawView.getIntervalTime()) : "", buildToServerData2, this.mLineDrawView.isSmoothLine() ? LineDataUtil.buildToServerData(this.mLineDrawView.getStrSmoothData(), this.mLineDrawView.getIntervalTime()) : "", buildToServerData) : LineDataUtil.buildToServerStructure(1, buildToServerData, buildToServerData2);
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        initData();
        initProductData();
        this.mImgSetting.setmSMSType(this.mModeType);
        this.mLineDrawView.setmType(this.mModeType);
        int status = BleManager.getInstance(this).getStatus();
        ImageView imageView = this.mTitleBluetooth;
        if (status == 2) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_connect));
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_disconnect));
        }
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerMode.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        PlayModeListAdapter playModeListAdapter = new PlayModeListAdapter(this, this.mList);
        this.mAdapter = playModeListAdapter;
        this.mRecyclerMode.setAdapter(playModeListAdapter);
        initActivityResult();
        refreshBtn();
        makeFilter();
        this.mRefreshLayoutMode.autoRefresh();
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.17
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                if (activityResult.getResultCode() == -1) {
                    LogUtil.e("蓝牙打开");
                    PlayModeCreateActivity2.this.startActivity(new Intent(PlayModeCreateActivity2.this, (Class<?>) BluetootListActivity.class));
                } else if (activityResult.getResultCode() == 0) {
                    LogUtil.e("蓝牙打不开");
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2, String str3) {
        if (TextUtils.isEmpty(str2) || TextUtils.isEmpty(str3)) {
            return;
        }
        str2.hashCode();
        if (str2.equals(Constans.piuCommand)) {
            int intValue = Integer.valueOf(str3, 16).intValue();
            Message message = new Message();
            message.what = 6;
            message.obj = Integer.valueOf(intValue);
            this.mHandler.sendMessage(message);
        }
    }

    private void initData() {
        try {
            if (getIntent() != null) {
                this.action = getIntent().getStringExtra(NativeProtocol.WEB_DIALOG_ACTION);
                PlayModeModel playModeModel = (PlayModeModel) getIntent().getParcelableExtra("editMode");
                this.mEditMode = playModeModel;
                if (playModeModel == null) {
                    if (ProductUtil.isSMSType()) {
                        this.mModeType = 2;
                        return;
                    } else {
                        this.mModeType = 1;
                        return;
                    }
                }
                try {
                    String[] buildServerToClientData = LineDataUtil.buildServerToClientData(playModeModel.getModePoints());
                    if (!TextUtils.isEmpty(buildServerToClientData[0])) {
                        int parseInt = Integer.parseInt(buildServerToClientData[0]);
                        for (int i = 1; i < buildServerToClientData.length; i++) {
                            buildServerToClientData[i] = LineDataUtil.buildToUIData(buildServerToClientData[i], this.mLineDrawView.getIntervalTime(), parseInt);
                            LogUtil.e("modePointArr[" + i + "] ： " + buildServerToClientData[i].toString());
                        }
                        if (buildServerToClientData.length == 5) {
                            this.mModeType = 2;
                            this.mLineDrawView.addData(buildServerToClientData[4], buildServerToClientData[2], buildServerToClientData[1], buildServerToClientData[3]);
                        } else {
                            this.mModeType = 1;
                            this.mLineDrawView.addData(buildServerToClientData[1], buildServerToClientData[2], "", "");
                        }
                    }
                    if (TextUtils.isEmpty(this.mEditMode.getModeSetting())) {
                        return;
                    }
                    this.mEditMode.initModeSettingsModel();
                    if (this.mEditMode.getModeSettingsModel() == null || MyApplication.getApplication().getmProductModel() == null || !Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
                        return;
                    }
                    if (BleManager.getInstance(this).getmConnBleDevice().getStartPosition() != this.mEditMode.getModeSettingsModel().getStrokeMin()) {
                        this.mHandler.sendEmptyMessageDelayed(3, this.delayMillis);
                        this.delayMillis += 200;
                    }
                    if (BleManager.getInstance(this).getmConnBleDevice().getStopPosition() != this.mEditMode.getModeSettingsModel().getStrokeMax()) {
                        this.mHandler.sendEmptyMessageDelayed(4, this.delayMillis);
                        this.delayMillis += 200;
                    }
                    if (BleManager.getInstance(this).getmConnBleDevice().getSmoothness() != this.mEditMode.getModeSettingsModel().getSmooth()) {
                        this.mHandler.sendEmptyMessageDelayed(5, this.delayMillis);
                        this.delayMillis += 200;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initProductData() {
        this.layout_mark.setVisibility(8);
        this.slider_position.setVisibility(8);
        this.slider_smooth.setVisibility(8);
        if (ProductUtil.isSMVType()) {
            this.mLineDrawView.setVibrateLine(true);
            this.mLineDrawView.setStartPositionLine(false);
            this.mLineDrawView.setSmoothLine(false);
            this.mTouchLayout1.setVisibility(0);
            this.mTouchBgLayout.setVisibility(0);
            this.mTouchBgLayout1.setVisibility(0);
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.mTouchLayout.getLayoutParams();
            layoutParams.width = MeasureUtil.dip2px(this, 70.0f);
            this.mTouchLayout.setLayoutParams(layoutParams);
            this.mImgTouch.setImageResource(R.mipmap.ic_speed_button);
        } else if (ProductUtil.isSMSType()) {
            int i = this.mModeType;
            LineDrawView5 lineDrawView5 = this.mLineDrawView;
            if (i == 2) {
                lineDrawView5.setVibrateLine(true);
                this.mLineDrawView.setStartPositionLine(true);
                this.mLineDrawView.setSmoothLine(true);
                this.layout_mark.setVisibility(0);
                this.slider_position.setVisibility(0);
                this.slider_smooth.setVisibility(0);
                int startPosition = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
                int stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
                int smoothness = BleManager.getInstance(this).getmConnBleDevice().getSmoothness();
                this.slider_position.setProgress(startPosition, stopPosition);
                this.slider_smooth.setValue(smoothness);
                this.mLineDrawView.setmStartPositionScale(startPosition);
                this.mLineDrawView.setmCurrentVibrateScale(stopPosition);
                this.mLineDrawView.setmSmoothScale(smoothness);
            } else {
                lineDrawView5.setVibrateLine(false);
                this.mLineDrawView.setStartPositionLine(false);
                this.mLineDrawView.setSmoothLine(false);
                BleManager.getInstance(this).writeRunManual();
            }
            this.mTouchLayout1.setVisibility(8);
            this.mTouchBgLayout.setVisibility(8);
            this.mTouchBgLayout1.setVisibility(8);
            ConstraintLayout.LayoutParams layoutParams2 = (ConstraintLayout.LayoutParams) this.mTouchLayout.getLayoutParams();
            layoutParams2.width = -1;
            this.mTouchLayout.setLayoutParams(layoutParams2);
            this.mImgTouch.setImageResource(R.mipmap.ic_speed_button_single);
        } else {
            this.mLineDrawView.setVibrateLine(false);
            this.mLineDrawView.setStartPositionLine(false);
            this.mLineDrawView.setSmoothLine(false);
            this.mTouchLayout1.setVisibility(8);
            this.mTouchBgLayout.setVisibility(8);
            this.mTouchBgLayout1.setVisibility(8);
            ConstraintLayout.LayoutParams layoutParams3 = (ConstraintLayout.LayoutParams) this.mTouchLayout.getLayoutParams();
            layoutParams3.width = -1;
            this.mTouchLayout.setLayoutParams(layoutParams3);
            this.mImgTouch.setImageResource(R.mipmap.ic_speed_button_single);
        }
        if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getReversal() != 1) {
            this.img_reversal.clearAnimation();
            this.img_reversal.setVisibility(8);
            return;
        }
        this.img_reversal.clearAnimation();
        this.img_reversal.setVisibility(0);
        int reversal = BleManager.getInstance(this).getmConnBleDevice().getReversal();
        this.lastReversal = reversal;
        ImageView imageView = this.img_reversal;
        if (reversal == 1) {
            AnimUtil.rotate(imageView, 1);
            this.img_reversal.setRotationY(180.0f);
        } else {
            AnimUtil.rotate(imageView, 0);
            this.img_reversal.setRotationY(0.0f);
        }
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleBluetooth = (ImageView) findViewById(R.id.title_img_bluetooth);
        this.mLineDrawView = (LineDrawView5) findViewById(R.id.lineDrawView5);
        this.mTxtStartTime = (TextView) findViewById(R.id.txt_start_time);
        this.mTxtEndTime = (TextView) findViewById(R.id.txt_end_time);
        this.mSeekTime = (SeekBar) findViewById(R.id.seek_time);
        this.mImgModeStart = (ImageView) findViewById(R.id.img_mode_start);
        this.mImgModeStop = (ImageView) findViewById(R.id.img_mode_stop);
        this.mImgModePlay = (ImageView) findViewById(R.id.img_mode_play);
        this.mImgModeSelect = (ImageView) findViewById(R.id.img_mode_select);
        this.mImgModeCopy = (ImageView) findViewById(R.id.img_mode_copy);
        this.mImgModePaste = (ImageView) findViewById(R.id.img_mode_paste);
        this.mImgModeDel = (ImageView) findViewById(R.id.img_mode_del);
        this.mImgModeList = (ImageView) findViewById(R.id.img_mode_list);
        this.mBtnSave = (TextView) findViewById(R.id.btn_save);
        this.mLayoutModeList = (LinearLayout) findViewById(R.id.layout_mode_list);
        this.mTxtModeList = (TextView) findViewById(R.id.txt_mode_list);
        this.mRefreshLayoutMode = (SmartRefreshLayout) findViewById(R.id.refreshLayout_mode);
        this.mRecyclerMode = (RecyclerView) findViewById(R.id.recycler_mode);
        this.mLayoutTouch = (FrameLayout) findViewById(R.id.layout_touch);
        this.mTouchBgLayout = (FrameLayout) findViewById(R.id.layout_bg);
        this.mTouchBgLayout1 = (FrameLayout) findViewById(R.id.layout_bg1);
        this.mTouchLayout = (TouchLayout) findViewById(R.id.touch_layout);
        this.mTouchLayout1 = (TouchLayout) findViewById(R.id.touch_layout1);
        this.mImgTouch = (ImageView) findViewById(R.id.img_touch);
        this.mLayoutArrow = (LinearLayout) findViewById(R.id.layout_arrow);
        this.mArrowBoth = (BothWayArrowView) findViewById(R.id.arrow_both);
        this.mImgGesture = (ImageView) findViewById(R.id.img_gesture);
        this.img_reversal = (ImageView) findViewById(R.id.img_reversal);
        this.img_piu = (ImageView) findViewById(R.id.img_piu);
        this.mImgSetting = (PlaySettingImageView) findViewById(R.id.img_mode_setting);
        this.layout_mark = (LinearLayout) findViewById(R.id.layout_mark);
        this.txt_mode_speed = (TextView) findViewById(R.id.txt_mode_speed);
        this.txt_mode_max_stroke = (TextView) findViewById(R.id.txt_mode_max_stroke);
        this.txt_mode_depth = (TextView) findViewById(R.id.txt_mode_depth);
        this.txt_mode_smooth = (TextView) findViewById(R.id.txt_mode_smooth);
        this.slider_position = (DoubleSliderView) findViewById(R.id.slider_position);
        this.slider_smooth = (NodeSliderView) findViewById(R.id.slider_smooth);
        LanguageUtil.setText(this, this.txt_mode_speed, "speed");
        LanguageUtil.setText(this, this.txt_mode_max_stroke, "maxStroke");
        LanguageUtil.setText(this, this.txt_mode_depth, "Depth");
        LanguageUtil.setText(this, this.txt_mode_smooth, "smoothness");
        LanguageUtil.setText(this, this.mTitleName, "addMode");
        LanguageUtil.setText(this, this.mTxtModeList, "selectSnippet");
        LanguageUtil.setText(this, this.mBtnSave, "complete");
        this.mTxtStartTime.setText("00:00");
        this.mTxtEndTime.setText("00:00");
        this.mSeekTime.setMax(0);
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_CONN_STATE);
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeBlock(String str) {
        this.baseRequest.getApi().playModeBlock((String) SharedPreferencesUtil.get(this, Constans.Token), str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ModeBlockModel>(this, null, false) { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.20
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PlayModeCreateActivity2.this.mRefreshLayoutMode.finishRefresh();
                PlayModeCreateActivity2.this.mRefreshLayoutMode.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                if (PlayModeCreateActivity2.this.mRefreshLayoutMode != null) {
                    PlayModeCreateActivity2.this.mRefreshLayoutMode.finishRefresh();
                    PlayModeCreateActivity2.this.mRefreshLayoutMode.finishLoadMore();
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ModeBlockModel modeBlockModel) {
                if (modeBlockModel == null || modeBlockModel.getSuccess() != 1) {
                    return;
                }
                if (TextUtils.isEmpty(PlayModeCreateActivity2.this.mRequestId)) {
                    PlayModeCreateActivity2.this.mList.clear();
                }
                if (!TextUtils.isEmpty(modeBlockModel.getRequestId())) {
                    PlayModeCreateActivity2.this.mRequestId = modeBlockModel.getRequestId();
                }
                if (modeBlockModel.getModeBlocks() != null && modeBlockModel.getModeBlocks().length > 0) {
                    PlayModeCreateActivity2.this.mList.addAll(Arrays.asList(modeBlockModel.getModeBlocks()));
                }
                if (PlayModeCreateActivity2.this.mAdapter != null) {
                    PlayModeCreateActivity2.this.mAdapter.notifyDataSetChanged();
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeSave(String str, String str2, String str3) {
        String str4;
        String str5 = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        if (TextUtils.isEmpty(deviceCode)) {
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "deviceNotConnect"));
            return;
        }
        if (!ProductUtil.hasSType()) {
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "deviceNotCreateMode"));
            return;
        }
        boolean z = true;
        if (ProductUtil.isSMSType() && this.mModeType == 1 && BleManager.getInstance(this).getmConnBleDevice() != null) {
            try {
                str4 = this.gson.toJson(new ModeSettingModel(BleManager.getInstance(this).getmConnBleDevice().getStartPosition(), BleManager.getInstance(this).getmConnBleDevice().getStopPosition(), BleManager.getInstance(this).getmConnBleDevice().getSmoothness()));
            } catch (Exception e) {
                e.printStackTrace();
            }
            String str6 = str4;
            LogUtil.e("setting:" + str6);
            this.baseRequest.getApi().playModeSave(str5, str, str2, deviceCode, str6, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeCreateModel>(this, this.progressDialog, z) { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.19
                /* JADX INFO: Access modifiers changed from: protected */
                @Override // com.hismith.hismithapp.http.RxSubscribe
                public void onSuccess(PlayModeCreateModel playModeCreateModel) {
                    if (playModeCreateModel.getSuccess() == 1) {
                        ToastUtil.showShortToastCenter(PlayModeCreateActivity2.this, LanguageUtil.getLanguageByKey(PlayModeCreateActivity2.this, "saveed"));
                        Intent intent = new Intent();
                        intent.putExtra("modePoint", playModeCreateModel.getModePoints());
                        intent.putExtra("modeSetting", playModeCreateModel.getModeSetting());
                        PlayModeCreateActivity2.this.setResult(17, intent);
                        PlayModeCreateActivity2.this.finish();
                    }
                }
            });
        }
        str4 = null;
        String str62 = str4;
        LogUtil.e("setting:" + str62);
        this.baseRequest.getApi().playModeSave(str5, str, str2, deviceCode, str62, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeCreateModel>(this, this.progressDialog, z) { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.19
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeCreateModel playModeCreateModel) {
                if (playModeCreateModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(PlayModeCreateActivity2.this, LanguageUtil.getLanguageByKey(PlayModeCreateActivity2.this, "saveed"));
                    Intent intent = new Intent();
                    intent.putExtra("modePoint", playModeCreateModel.getModePoints());
                    intent.putExtra("modeSetting", playModeCreateModel.getModeSetting());
                    PlayModeCreateActivity2.this.setResult(17, intent);
                    PlayModeCreateActivity2.this.finish();
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshBtn() {
        boolean isRecord = this.mLineDrawView.isRecord();
        ImageView imageView = this.mImgModeStart;
        if (isRecord) {
            imageView.setImageResource(R.mipmap.btn_mode_edit_start_purple);
            this.mImgModeStart.setEnabled(false);
            this.mImgModeStop.setImageResource(R.mipmap.btn_mode_edit_stop_white);
            this.mImgModeStop.setEnabled(true);
            this.mImgModePlay.setImageResource(R.mipmap.btn_mode_edit_play_gray);
            this.mImgModePlay.setEnabled(false);
            this.mImgModeSelect.setImageResource(R.mipmap.btn_mode_edit_select_gray);
            this.mImgModeSelect.setEnabled(false);
        } else {
            imageView.setImageResource(R.mipmap.btn_mode_edit_start_white);
            this.mImgModeStart.setEnabled(true);
            this.mImgModeStop.setImageResource(R.mipmap.btn_mode_edit_stop_gray);
            this.mImgModeStop.setEnabled(false);
            if (!this.mLineDrawView.isDataEmpty()) {
                this.mImgModePlay.setImageResource(R.mipmap.btn_mode_edit_play_white);
                this.mImgModePlay.setEnabled(true);
                if (!this.mLineDrawView.isPlay()) {
                    this.mImgModeSelect.setImageResource(R.mipmap.btn_mode_edit_select_white);
                    this.mImgModeSelect.setEnabled(true);
                }
            }
            if (this.mLineDrawView.isPlay()) {
                this.mImgModeStart.setImageResource(R.mipmap.btn_mode_edit_start_gray);
                this.mImgModeStart.setEnabled(false);
                this.mImgModePlay.setImageResource(R.mipmap.btn_mode_edit_play_purple);
                this.mImgModePlay.setEnabled(true);
                this.mImgModeStop.setImageResource(R.mipmap.btn_mode_edit_stop_white);
                this.mImgModeStop.setEnabled(true);
                this.mImgModeSelect.setImageResource(R.mipmap.btn_mode_edit_select_gray);
                this.mImgModeSelect.setEnabled(false);
            } else {
                this.mImgModeStop.setImageResource(R.mipmap.btn_mode_edit_stop_gray);
                this.mImgModeStop.setEnabled(false);
            }
        }
        if (this.mLineDrawView.isDataEmpty()) {
            this.mImgModePlay.setImageResource(R.mipmap.btn_mode_edit_play_gray);
            this.mImgModePlay.setEnabled(false);
            this.mImgModeSelect.setImageResource(R.mipmap.btn_mode_edit_select_gray);
            this.mImgModeSelect.setEnabled(false);
        }
        if (this.mLineDrawView.isSelect()) {
            this.mImgModeStart.setImageResource(R.mipmap.btn_mode_edit_start_gray);
            this.mImgModeStart.setEnabled(false);
            this.mImgModePlay.setImageResource(R.mipmap.btn_mode_edit_play_gray);
            this.mImgModePlay.setEnabled(false);
            this.mImgModeSelect.setImageResource(R.mipmap.btn_mode_edit_select_purple);
            this.mImgModeSelect.setEnabled(true);
            this.mImgModeCopy.setImageResource(R.mipmap.btn_mode_edit_copy_white);
            this.mImgModeCopy.setEnabled(true);
            this.mImgModeDel.setImageResource(R.mipmap.btn_mode_edit_del_white);
            this.mImgModeDel.setEnabled(true);
        } else {
            if (this.mLineDrawView.isDataEmpty() || this.mLineDrawView.isRecord() || this.mLineDrawView.isPlay()) {
                this.mImgModeSelect.setImageResource(R.mipmap.btn_mode_edit_select_gray);
            } else {
                this.mImgModeSelect.setImageResource(R.mipmap.btn_mode_edit_select_white);
            }
            this.mImgModeCopy.setImageResource(R.mipmap.btn_mode_edit_copy_gray);
            this.mImgModeCopy.setEnabled(false);
            this.mImgModeDel.setImageResource(R.mipmap.btn_mode_edit_del_gray);
            this.mImgModeDel.setEnabled(false);
        }
        if (!this.mLineDrawView.isCopy()) {
            this.mImgModePaste.setImageResource(R.mipmap.btn_mode_edit_paste_gray);
            this.mImgModePaste.setEnabled(false);
        } else if (this.mLineDrawView.getIsPause() != 2 || this.mLineDrawView.isSelect()) {
            this.mImgModePaste.setImageResource(R.mipmap.btn_mode_edit_paste_gray);
            this.mImgModePaste.setEnabled(false);
        } else {
            this.mImgModePaste.setImageResource(R.mipmap.btn_mode_edit_paste_white);
            this.mImgModePaste.setEnabled(true);
        }
        boolean isSelectList = this.mLineDrawView.isSelectList();
        ImageView imageView2 = this.mImgModeList;
        if (isSelectList) {
            imageView2.setImageResource(R.mipmap.btn_mode_edit_list_purple);
            if (this.mLayoutModeList.getVisibility() != 0) {
                this.mLayoutModeList.setVisibility(0);
                this.mLayoutModeList.startAnimation(AnimationUtils.loadAnimation(this, R.anim.bottom_in));
                return;
            }
            return;
        }
        imageView2.setImageResource(R.mipmap.btn_mode_edit_list_white);
        if (this.mLayoutModeList.getVisibility() != 8) {
            this.mLayoutModeList.setVisibility(8);
            this.mLayoutModeList.startAnimation(AnimationUtils.loadAnimation(this, R.anim.bottom_out));
        }
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mTitleBluetooth.setOnClickListener(this);
        this.mImgModeStart.setOnClickListener(this);
        this.mImgModeStop.setOnClickListener(this);
        this.mImgModePlay.setOnClickListener(this);
        this.mImgModeSelect.setOnClickListener(this);
        this.mImgModeCopy.setOnClickListener(this);
        this.mImgModePaste.setOnClickListener(this);
        this.mImgModeDel.setOnClickListener(this);
        this.mImgModeList.setOnClickListener(this);
        this.mBtnSave.setOnClickListener(this);
        this.mRefreshLayoutMode.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.4
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                PlayModeCreateActivity2 playModeCreateActivity2 = PlayModeCreateActivity2.this;
                playModeCreateActivity2.playModeBlock(playModeCreateActivity2.mRequestId);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PlayModeCreateActivity2.this.mRequestId = null;
                PlayModeCreateActivity2.this.playModeBlock(null);
            }
        });
        this.mAdapter.setOnItemClickListener(new PlayModeListAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.5
            @Override // com.hismith.hismithapp.adapter.PlayModeListAdapter.OnItemClickListener
            public void onItemClick(int i) {
                try {
                    if (PlayModeCreateActivity2.this.mLineDrawView != null) {
                        String[] buildServerToClientData = LineDataUtil.buildServerToClientData(((ModeBlockModel.ModeBlock) PlayModeCreateActivity2.this.mList.get(i)).getBlockData());
                        if (!TextUtils.isEmpty(buildServerToClientData[0])) {
                            int parseInt = Integer.parseInt(buildServerToClientData[0]);
                            for (int i2 = 1; i2 < buildServerToClientData.length; i2++) {
                                buildServerToClientData[i2] = LineDataUtil.buildToUIData(buildServerToClientData[i2], PlayModeCreateActivity2.this.mLineDrawView.getIntervalTime(), parseInt);
                            }
                            int length = buildServerToClientData.length;
                            PlayModeCreateActivity2 playModeCreateActivity2 = PlayModeCreateActivity2.this;
                            if (length == 5) {
                                playModeCreateActivity2.mLineDrawView.addData(buildServerToClientData[4], buildServerToClientData[2], buildServerToClientData[1], buildServerToClientData[3]);
                            } else {
                                playModeCreateActivity2.mLineDrawView.addData(buildServerToClientData[1], buildServerToClientData[2], "", "");
                            }
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                LogUtil.e("onItemClick(" + i + "):" + ((ModeBlockModel.ModeBlock) PlayModeCreateActivity2.this.mList.get(i)).getBlockData());
            }
        });
        this.mTouchLayout.setOnScaleChangeListener(new TouchLayout.OnScaleChangeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.6
            @Override // com.hismith.hismithapp.view.TouchLayout.OnScaleChangeListener
            public void onScaleChange(int i) {
                PlayModeCreateActivity2.this.mLineDrawView.setmCurrentScale(i);
                if (PlayModeCreateActivity2.this.mLayoutArrow.getVisibility() != 8) {
                    PlayModeCreateActivity2.this.mLayoutArrow.setVisibility(8);
                    PlayModeCreateActivity2.this.mArrowBoth.setOnValueListener(null);
                }
            }
        });
        this.mTouchLayout1.setOnScaleChangeListener(new TouchLayout.OnScaleChangeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.7
            @Override // com.hismith.hismithapp.view.TouchLayout.OnScaleChangeListener
            public void onScaleChange(int i) {
                PlayModeCreateActivity2.this.mLineDrawView.setmCurrentVibrateScale(i);
                if (PlayModeCreateActivity2.this.mLayoutArrow.getVisibility() != 8) {
                    PlayModeCreateActivity2.this.mLayoutArrow.setVisibility(8);
                    PlayModeCreateActivity2.this.mArrowBoth.setOnValueListener(null);
                }
            }
        });
        this.mLineDrawView.setOnRecodeTimeListener(new OnRecodeTimeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.8
            @Override // com.hismith.hismithapp.surface_view.listener.OnRecodeTimeListener
            public void onRecodeTime(int i) {
                LogUtil.d("onRecodeTime time:" + i);
                PlayModeCreateActivity2.this.mSeekTime.setMax(i);
                PlayModeCreateActivity2.this.mTxtEndTime.setText(ModeTimeUtil.formatMilliTime(i));
                if (PlayModeCreateActivity2.this.mLineDrawView.isRecord()) {
                    if (!PlayModeCreateActivity2.this.isTouchSeek) {
                        PlayModeCreateActivity2.this.mSeekTime.setProgress(i);
                    }
                    PlayModeCreateActivity2.this.mTxtStartTime.setText(ModeTimeUtil.formatMilliTime(i));
                }
            }
        });
        this.mLineDrawView.setOnPlayTimeListener(new OnPlayTimeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.9
            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onComplete() {
                LogUtil.e("onComplete");
            }

            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onFunData(String str, String str2) {
                LogUtil.e("onFunData data0:" + str + ", data1:" + str2);
                int intValue = Integer.valueOf(str, 16).intValue();
                int intValue2 = Integer.valueOf(str2, 16).intValue();
                if (intValue == 254) {
                    if (intValue2 == 1) {
                        BleManager.getInstance(PlayModeCreateActivity2.this).writeOpenPiu();
                    }
                } else {
                    if (intValue != 255) {
                        return;
                    }
                    if (intValue2 == 0) {
                        AnimUtil.rotate(PlayModeCreateActivity2.this.img_reversal, 0);
                        PlayModeCreateActivity2.this.img_reversal.setRotationY(0.0f);
                        BleManager.getInstance(PlayModeCreateActivity2.this).writeCloseReversal();
                    } else if (intValue2 == 1) {
                        AnimUtil.rotate(PlayModeCreateActivity2.this.img_reversal, 1);
                        PlayModeCreateActivity2.this.img_reversal.setRotationY(180.0f);
                        BleManager.getInstance(PlayModeCreateActivity2.this).writeOpenReversal();
                    }
                }
            }

            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onLinePause() {
                if (ProductUtil.isSMSType() && PlayModeCreateActivity2.this.mModeType == 1) {
                    return;
                }
                if (PlayModeCreateActivity2.this.lastSpeed != 0) {
                    BleManager.getInstance(PlayModeCreateActivity2.this).writeSpeed(0);
                    PlayModeCreateActivity2.this.lastSpeed = 0;
                }
                if (PlayModeCreateActivity2.this.lastVibrate != 0) {
                    PlayModeCreateActivity2.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.9.4
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(PlayModeCreateActivity2.this).writeVibrate(0);
                            PlayModeCreateActivity2.this.lastVibrate = 0;
                        }
                    }, 100L);
                }
            }

            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onPlayTime(int i) {
                LogUtil.d("onPlayTime:" + i);
                if (!PlayModeCreateActivity2.this.isTouchSeek) {
                    PlayModeCreateActivity2.this.mSeekTime.setProgress(i);
                }
                PlayModeCreateActivity2.this.mTxtStartTime.setText(ModeTimeUtil.formatMilliTime(i));
            }

            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onPositionRangeChange(final int i, final int i2) {
                LogUtil.e("onPositionRangeChange start:" + i + ", stop:" + i2);
                int i3 = 15;
                int i4 = 30;
                if (i >= PlayModeCreateActivity2.this.lastStopPosition) {
                    i4 = 15;
                    i3 = 30;
                }
                if (i != PlayModeCreateActivity2.this.lastStartPosition && BleManager.getInstance(PlayModeCreateActivity2.this).getmConnBleDevice() != null) {
                    PlayModeCreateActivity2.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.9.1
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(PlayModeCreateActivity2.this).writeData(BleManager.getInstance(PlayModeCreateActivity2.this).getBleCCData(7, i));
                        }
                    }, i3);
                    PlayModeCreateActivity2.this.lastStartPosition = i;
                }
                if (i2 != PlayModeCreateActivity2.this.lastStopPosition && BleManager.getInstance(PlayModeCreateActivity2.this).getmConnBleDevice() != null) {
                    PlayModeCreateActivity2.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.9.2
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(PlayModeCreateActivity2.this).writeData(BleManager.getInstance(PlayModeCreateActivity2.this).getBleCCData(8, i2));
                        }
                    }, i4);
                    PlayModeCreateActivity2.this.lastStopPosition = i2;
                }
                if (ProductUtil.isSMSType() && PlayModeCreateActivity2.this.mModeType == 2 && PlayModeCreateActivity2.this.mLineDrawView.isPlay()) {
                    PlayModeCreateActivity2.this.slider_position.setProgress(i, i2);
                }
            }

            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onSmoothChange(final int i) {
                LogUtil.e("onSmoothChange smooth:" + i);
                if (i != PlayModeCreateActivity2.this.lastSmooth && BleManager.getInstance(PlayModeCreateActivity2.this).getmConnBleDevice() != null) {
                    PlayModeCreateActivity2.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.9.3
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(PlayModeCreateActivity2.this).writeData(BleManager.getInstance(PlayModeCreateActivity2.this).getBleCCData(9, i));
                        }
                    }, 45L);
                    PlayModeCreateActivity2.this.lastSmooth = i;
                }
                if (ProductUtil.isSMSType() && PlayModeCreateActivity2.this.mModeType == 2 && PlayModeCreateActivity2.this.mLineDrawView.isPlay()) {
                    PlayModeCreateActivity2.this.slider_smooth.setValue(i);
                }
            }

            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onSpeedChange(int i) {
                LogUtil.d("onSpeedChange:" + i);
                if (i == PlayModeCreateActivity2.this.lastSpeed || BleManager.getInstance(PlayModeCreateActivity2.this).getmConnBleDevice() == null) {
                    return;
                }
                if (ProductUtil.isSMSType() && PlayModeCreateActivity2.this.mModeType == 1) {
                    BleManager.getInstance(PlayModeCreateActivity2.this).writePosition(i);
                } else {
                    BleManager.getInstance(PlayModeCreateActivity2.this).writeSpeed(i);
                }
                PlayModeCreateActivity2.this.lastSpeed = i;
            }

            @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
            public void onVibrateChange(int i) {
                if (i == PlayModeCreateActivity2.this.lastVibrate || BleManager.getInstance(PlayModeCreateActivity2.this).getmConnBleDevice() == null) {
                    return;
                }
                BleManager.getInstance(PlayModeCreateActivity2.this).writeVibrate(i);
                PlayModeCreateActivity2.this.lastVibrate = i;
            }
        });
        this.mLineDrawView.setOnSelectTimeListener(new OnSelectTimeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.10
            @Override // com.hismith.hismithapp.surface_view.listener.OnSelectTimeListener
            public void onSelectTime(int i, int i2) {
                LogUtil.e("onSelectTime() startTime：" + i + ", endTime：" + i2);
            }
        });
        this.mLineDrawView.setOnStatusChangeListener(new OnStatusChangeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.11
            @Override // com.hismith.hismithapp.surface_view.listener.OnStatusChangeListener
            public void OnStatusChange() {
                PlayModeCreateActivity2.this.refreshBtn();
            }
        });
        this.mSeekTime.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.12
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                PlayModeCreateActivity2.this.isTouchSeek = true;
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                PlayModeCreateActivity2.this.isTouchSeek = false;
                PlayModeCreateActivity2.this.mLineDrawView.setPlayTime(seekBar.getProgress());
            }
        });
        this.mArrowBoth.setOnValueListener(new BothWayArrowView.OnValueListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.13
            @Override // com.hismith.hismithapp.view.BothWayArrowView.OnValueListener
            public void onValueChange(float f) {
                PlayModeCreateActivity2.this.mImgGesture.setRotation(f * (-45.0f));
            }
        });
        this.img_reversal.setOnClickListener(this);
        this.img_piu.setOnTouchListener(this.piuOnTouchListener);
        this.slider_position.setOnRangeChangeListener(new DoubleSliderView.OnRangeChangeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.14
            @Override // com.hismith.hismithapp.view.DoubleSliderView.OnRangeChangeListener
            public void onMaxValueChanged(int i) {
                LogUtil.e("onMaxValueChanged ：" + i);
                PlayModeCreateActivity2.this.mLineDrawView.setmCurrentVibrateScale(i);
            }

            @Override // com.hismith.hismithapp.view.DoubleSliderView.OnRangeChangeListener
            public void onMinValueChanged(int i) {
                LogUtil.e("onMinValueChanged ：" + i);
                PlayModeCreateActivity2.this.mLineDrawView.setmStartPositionScale(i);
            }

            @Override // com.hismith.hismithapp.view.DoubleSliderView.OnRangeChangeListener
            public void onRangeChangeEnd() {
                LogUtil.e("onRangeChangeEnd ");
            }

            @Override // com.hismith.hismithapp.view.DoubleSliderView.OnRangeChangeListener
            public void onRangeChangeStart() {
                LogUtil.e("onRangeChangeStart ");
            }
        });
        this.slider_smooth.setOnValueChangeListener(new NodeSliderView.OnValueChangeListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.15
            @Override // com.hismith.hismithapp.view.NodeSliderView.OnValueChangeListener
            public void onValueChanged(int i) {
                LogUtil.e("onValueChanged ：" + i);
                PlayModeCreateActivity2.this.mLineDrawView.setmSmoothScale(i);
            }
        });
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            onBackPressed();
            return;
        }
        if (id == R.id.title_img_bluetooth) {
            if (bluetoothIsEnabled()) {
                startActivity(new Intent(this, (Class<?>) BluetootListActivity.class));
                return;
            }
            return;
        }
        if (id == R.id.img_mode_start) {
            this.mLineDrawView.recode();
            return;
        }
        if (id == R.id.img_mode_stop) {
            this.mLineDrawView.pause();
            return;
        }
        if (id == R.id.img_mode_play) {
            this.mLineDrawView.play();
            return;
        }
        int i = 0;
        if (id == R.id.img_mode_select) {
            this.mLineDrawView.select(!r7.isSelect());
            if (this.mLineDrawView.isSelect() && this.mLineDrawView.isSelectList()) {
                this.mLineDrawView.selectList(false);
                return;
            }
            return;
        }
        if (id == R.id.img_mode_copy) {
            this.mLineDrawView.copy();
            return;
        }
        if (id == R.id.img_mode_paste) {
            this.mLineDrawView.paste();
            return;
        }
        if (id == R.id.img_mode_del) {
            this.mLineDrawView.delete();
            return;
        }
        if (id == R.id.img_mode_list) {
            boolean isSelectList = this.mLineDrawView.isSelectList();
            LineDrawView5 lineDrawView5 = this.mLineDrawView;
            if (isSelectList) {
                lineDrawView5.selectList(false);
                return;
            } else {
                lineDrawView5.selectList(true);
                return;
            }
        }
        if (id != R.id.btn_save) {
            if (id == R.id.img_reversal) {
                if (this.lastReversal == 1) {
                    this.lastReversal = 0;
                } else {
                    this.lastReversal = 1;
                    i = 1;
                }
                this.mLineDrawView.addFunData(255, i);
                return;
            }
            return;
        }
        if (this.mLineDrawView.isRecord()) {
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "pleaseStopRecord"));
            return;
        }
        if (this.mLineDrawView.isDataEmpty()) {
            return;
        }
        if (this.mEditMode == null || !"edit".equals(this.action)) {
            DialogUtil.showEditDialog2(this, LanguageUtil.getLanguageByKey(this, "modeName"), "", null, null, new DialogUtil.OnEditContentCallback() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.21
                @Override // com.hismith.hismithapp.utils.DialogUtil.OnEditContentCallback
                public void onEditContent(String str) {
                    boolean isEmpty = TextUtils.isEmpty(str);
                    PlayModeCreateActivity2 playModeCreateActivity2 = PlayModeCreateActivity2.this;
                    if (isEmpty) {
                        ToastUtil.showShortToastCenter(playModeCreateActivity2, LanguageUtil.getLanguageByKey(playModeCreateActivity2, "pleaseEnterModeName"));
                    } else {
                        playModeCreateActivity2.playModeSave(null, str, playModeCreateActivity2.getModeData());
                    }
                }
            });
        } else {
            playModeSave(this.mEditMode.getModeId() + "", this.mEditMode.getModeName(), getModeData());
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                if (PlayModeCreateActivity2.this.mLineDrawView == null || PlayModeCreateActivity2.this.mLineDrawView.isDataEmpty()) {
                    PlayModeCreateActivity2.this.finish();
                } else {
                    DialogUtil.showConfirmDialog(PlayModeCreateActivity2.this, "prompt", "unsavedData", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.2.1
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            DialogUtil.dismissDialog();
                            PlayModeCreateActivity2.this.finish();
                        }
                    });
                }
            }
        });
        setContentView(R.layout.activity_play_mode_create2);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModeCreateActivity2.3
            @Override // java.lang.Runnable
            public void run() {
                BleManager.getInstance(PlayModeCreateActivity2.this).writeShutdown();
            }
        }, 100L);
        unRegisterReceiver();
    }

    public void setCanEdit(View view) {
        view.setFocusable(false);
        view.setFocusableInTouchMode(false);
    }

    public void setCanNotEdit(View view) {
        view.setFocusable(false);
        view.setFocusableInTouchMode(false);
    }
}
