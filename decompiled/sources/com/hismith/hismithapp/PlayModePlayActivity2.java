package com.hismith.hismithapp;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.Display;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;
import androidx.core.view.ViewCompat;
import androidx.fragment.app.FragmentTransaction;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.facebook.internal.NativeProtocol;
import com.hismith.hismithapp.adapter.PlayModeColorAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.danmu.JsonDanmakuLoader;
import com.hismith.hismithapp.danmu.JsonDanmakuParser;
import com.hismith.hismithapp.dialog.RenameDialog;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.PiuTimerManager;
import com.hismith.hismithapp.model.BarrageModel;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.PlayModeActionModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.popup.ActionAttachPopup2;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.surface_view.LineDataUtil;
import com.hismith.hismithapp.surface_view.LineDrawView5;
import com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.ModeTimeUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StringUtils;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.RightDraggableFrameLayout;
import com.hismith.hismithapp.view.plug.PlugPiuTimer;
import com.lxj.xpopup.util.KeyboardUtils;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import master.flame.danmaku.controller.DrawHandler;
import master.flame.danmaku.danmaku.model.BaseDanmaku;
import master.flame.danmaku.danmaku.model.DanmakuTimer;
import master.flame.danmaku.danmaku.model.android.DanmakuContext;
import master.flame.danmaku.danmaku.parser.BaseDanmakuParser;
import master.flame.danmaku.ui.widget.DanmakuView;

/* loaded from: classes3.dex */
public class PlayModePlayActivity2 extends BaseActivity implements View.OnClickListener, OnPlayTimeListener, SeekBar.OnSeekBarChangeListener, SensorEventListener {
    private static final int WHAT_PLAY_MODE_AUTO = 6;
    private static final int WHAT_PLAY_MODE_MANUAL = 2;
    private static final int WHAT_PLAY_MODE_SMOOTH = 5;
    private static final int WHAT_PLAY_MODE_START = 1;
    private static final int WHAT_PLAY_MODE_START_PIS = 3;
    private static final int WHAT_PLAY_MODE_STOP_PIS = 4;
    private RetrofitRequest baseRequest;
    private DanmakuContext danmakuContext;
    private ImageView img_auto_play;
    private ImageView img_mode_play_piu;
    private int lastSmooth;
    private int lastSpeed;
    private int lastStartPosition;
    private int lastStopPosition;
    private int lastVibrate;
    private RightDraggableFrameLayout layout_fragment_piu;
    private LinearLayout layout_mark;
    private PlayModeColorAdapter mAdapter;
    private DanmakuView mDanmakuView;
    private EditText mEditBarrageContent;
    private ImageView mImgBarrage;
    private ImageView mImgBarrageSend;
    private ImageView mImgCollectCount;
    private ImageView mImgLikeCount;
    private ImageView mImgMore;
    private ImageView mImgPlay;
    private ImageView mImgPlayCount;
    private ImageView mImgType;
    private LayoutInflater mInflater;
    private ConstraintLayout mLayoutBarrage;
    private FrameLayout mLayoutBarrageColor;
    private GridLayoutManager mLayoutManager;
    private ConstraintLayout mLayoutSetting;
    private LineDrawView5 mLineDrawView;
    private BaseDanmakuParser mParser;
    private PlayModeModel mPlayModeModel;
    private PlugPiuTimer mPlugPiuTimer;
    private RecyclerView mRecyclerColor;
    private SeekBar mSeekBar;
    private Sensor mSensor;
    private SensorManager mSensorManager;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TextView mTxtCollectCount;
    private TextView mTxtEndTime;
    private TextView mTxtLikeCount;
    private TextView mTxtPlayCount;
    private TextView mTxtStartTime;
    private View mViewBarrageColor;
    private TextView txt_mode_depth;
    private TextView txt_mode_max_stroke;
    private TextView txt_mode_smooth;
    private TextView txt_mode_speed;
    private TextView txt_setting_position_max;
    private TextView txt_setting_position_min;
    private TextView txt_setting_smooth;
    private int mSMSModeType = 1;
    private boolean isBarrageShow = true;
    private boolean isPrepared = false;
    private boolean isPlay = false;
    private boolean isDownSeek = false;
    private boolean isSendPlay = false;
    private int mPlayType = 0;
    private boolean isAutoPlay = false;
    private boolean isShowPiuUI = true;
    private boolean mModeChanged = false;
    private final int normalDelay = 0;
    private int delayMillis = 0;
    Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            switch (message.what) {
                case 1:
                    if (PlayModePlayActivity2.this.isPrepared) {
                        PlayModePlayActivity2.this.mLineDrawView.play();
                        PlayModePlayActivity2.this.isPlay = true;
                    }
                    boolean z = PlayModePlayActivity2.this.isPlay;
                    PlayModePlayActivity2 playModePlayActivity2 = PlayModePlayActivity2.this;
                    if (!z) {
                        playModePlayActivity2.mImgPlay.setImageDrawable(ContextCompat.getDrawable(PlayModePlayActivity2.this, R.mipmap.ic_mode_play));
                        break;
                    } else {
                        playModePlayActivity2.mImgPlay.setImageDrawable(ContextCompat.getDrawable(PlayModePlayActivity2.this, R.mipmap.ic_mode_pause));
                        break;
                    }
                case 2:
                    BleManager.getInstance(PlayModePlayActivity2.this).writeRunManual();
                    break;
                case 3:
                    if (PlayModePlayActivity2.this.mPlayModeModel != null && PlayModePlayActivity2.this.mPlayModeModel.getModeSettingsModel() != null) {
                        BleManager.getInstance(PlayModePlayActivity2.this).writeStartPosition(PlayModePlayActivity2.this.mPlayModeModel.getModeSettingsModel().getStrokeMin());
                        break;
                    }
                    break;
                case 4:
                    if (PlayModePlayActivity2.this.mPlayModeModel != null && PlayModePlayActivity2.this.mPlayModeModel.getModeSettingsModel() != null) {
                        BleManager.getInstance(PlayModePlayActivity2.this).writeStopPosition(PlayModePlayActivity2.this.mPlayModeModel.getModeSettingsModel().getStrokeMax());
                        break;
                    }
                    break;
                case 5:
                    if (PlayModePlayActivity2.this.mPlayModeModel != null && PlayModePlayActivity2.this.mPlayModeModel.getModeSettingsModel() != null) {
                        BleManager.getInstance(PlayModePlayActivity2.this).writeSmooth(PlayModePlayActivity2.this.mPlayModeModel.getModeSettingsModel().getSmooth());
                        break;
                    }
                    break;
                case 6:
                    BleManager.getInstance(PlayModePlayActivity2.this).writeRunAuto();
                    break;
            }
            return false;
        }
    });
    private final int delay = 15;
    int defDelayMillils = 200;
    private boolean isReversal = false;
    private int reversalCount = 0;
    private int recoverCount = 0;

    /* JADX INFO: Access modifiers changed from: private */
    public void addDanmaku(String str, String str2) {
        BaseDanmaku createDanmaku = this.danmakuContext.mDanmakuFactory.createDanmaku(1, this.danmakuContext);
        if (createDanmaku == null || this.mDanmakuView == null) {
            return;
        }
        int intValue = Integer.valueOf(str2, 16).intValue();
        int i = ViewCompat.MEASURED_STATE_MASK;
        int i2 = intValue | ViewCompat.MEASURED_STATE_MASK;
        createDanmaku.text = str;
        createDanmaku.padding = 5;
        createDanmaku.priority = (byte) 1;
        createDanmaku.isLive = false;
        createDanmaku.setTime(this.mDanmakuView.getCurrentTime());
        createDanmaku.textSize = this.mTxtStartTime.getTextSize();
        createDanmaku.textColor = i2;
        if (i2 <= -16777216) {
            i = -1;
        }
        createDanmaku.textShadowColor = i;
        createDanmaku.borderColor = -1;
        this.mDanmakuView.addDanmaku(createDanmaku);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public float autosizeText(float f) {
        return f / (getResources().getDisplayMetrics().density + 0.2f);
    }

    private void change60RefreshRate() {
        try {
            Display.Mode mode = getWindowManager().getDefaultDisplay().getSupportedModes()[0];
            for (Display.Mode mode2 : getWindowManager().getDefaultDisplay().getSupportedModes()) {
                LogUtil.e("帧率：" + mode2.getRefreshRate());
                if (mode2.getRefreshRate() >= 60.0f && mode2.getRefreshRate() < mode.getRefreshRate()) {
                    LogUtil.e("换成帧率：" + mode2.getRefreshRate());
                    getWindow().getAttributes().preferredDisplayModeId = mode2.getModeId();
                    return;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(int i, String str, String str2) {
        this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(i), "playMode", str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.PlayModePlayActivity2.14
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(PlayModePlayActivity2.this, LanguageUtil.getLanguageByKey(PlayModePlayActivity2.this, "reportedSuccess"));
                }
            }
        });
    }

    private BaseDanmakuParser createParser(BarrageModel barrageModel) {
        JsonDanmakuLoader instance = JsonDanmakuLoader.instance();
        instance.load(barrageModel);
        JsonDanmakuParser jsonDanmakuParser = new JsonDanmakuParser(this, this.mTxtStartTime.getTextSize());
        jsonDanmakuParser.load(instance.getDataSource());
        return jsonDanmakuParser;
    }

    private List<ActionAttachPopup2.ActionItem2> getActionList() {
        ArrayList arrayList = new ArrayList();
        PlayModeModel playModeModel = this.mPlayModeModel;
        if (playModeModel != null) {
            if (playModeModel.getShared() == 1 && MyApplication.getApplication().getmUserModel() != null && this.mPlayModeModel.getUserId() != MyApplication.getApplication().getmUserModel().getUserId()) {
                if (this.mPlayModeModel.getCollected() == 1) {
                    arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_delete, "uncollect"));
                } else {
                    arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_add, "addCollect"));
                }
            }
            if (this.mPlayModeModel.getShared() == 0 && MyApplication.getApplication().getmUserModel() != null && this.mPlayModeModel.getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_rename, "rename"));
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_edit, "edit"));
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_share, "modeShare"));
            }
            if (this.mPlayModeModel.getUserKing() == 1 || (MyApplication.getApplication().getmUserModel() != null && this.mPlayModeModel.getUserId() == MyApplication.getApplication().getmUserModel().getUserId())) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_copy, "copy"));
            }
            if (MyApplication.getApplication().getmUserModel() != null && this.mPlayModeModel.getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_delete, "delete"));
            }
            if (this.mPlayModeModel.getShared() == 1) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_report, "report"));
            }
        }
        return arrayList;
    }

    private void getPlayModeBarrage() {
        if (this.mPlayModeModel == null) {
            return;
        }
        this.baseRequest.getApi().getPlayModeBarrage((String) SharedPreferencesUtil.get(this, Constans.Token), this.mPlayModeModel.getModeId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<BarrageModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.PlayModePlayActivity2.15
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(BarrageModel barrageModel) {
                if (barrageModel.getSuccess() == 1) {
                    PlayModePlayActivity2.this.prepareDanmu(barrageModel);
                }
            }
        });
    }

    private String getSelectedColor() {
        try {
            return Integer.toHexString(((ColorDrawable) this.mViewBarrageColor.getBackground()).getColor() & 16777215);
        } catch (Exception e) {
            e.printStackTrace();
            return "000000";
        }
    }

    private void hide() {
        this.mLayoutBarrageColor.setVisibility(8);
        this.mEditBarrageContent.setVisibility(8);
        this.mImgBarrageSend.setVisibility(8);
    }

    private void init() {
        change60RefreshRate();
        this.baseRequest = new RetrofitRequest();
        this.mInflater = LayoutInflater.from(this);
        if (getIntent() != null) {
            this.mPlayModeModel = (PlayModeModel) getIntent().getParcelableExtra("playModeModel");
        }
        boolean booleanValue = ((Boolean) SharedPreferencesUtil.get(this, Constans.ModePlayAutoPlay, false)).booleanValue();
        this.isAutoPlay = booleanValue;
        ImageView imageView = this.img_auto_play;
        if (booleanValue) {
            imageView.setImageResource(R.mipmap.ic_auto_switch_on);
        } else {
            imageView.setImageResource(R.mipmap.ic_auto_switch_off);
        }
        boolean z = this.isShowPiuUI;
        ImageView imageView2 = this.img_mode_play_piu;
        if (z) {
            imageView2.setImageResource(R.mipmap.ic_piu_switch_on);
        } else {
            imageView2.setImageResource(R.mipmap.ic_piu_switch_off);
        }
        initSensor();
        initModePoint(this.isAutoPlay);
        initRecyclerColor();
        initDanmu();
        getPlayModeBarrage();
    }

    private void initDanmu() {
        HashMap hashMap = new HashMap();
        hashMap.put(1, 5);
        HashMap hashMap2 = new HashMap();
        hashMap2.put(1, true);
        hashMap2.put(5, true);
        DanmakuContext create = DanmakuContext.create();
        this.danmakuContext = create;
        create.setDanmakuStyle(2, 3.0f).setDuplicateMergingEnabled(false).setScrollSpeedFactor(1.2f).setScaleTextSize(1.2f).setMaximumLines(hashMap).preventOverlapping(hashMap2);
        this.mDanmakuView.setCallback(new DrawHandler.Callback() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.3
            @Override // master.flame.danmaku.controller.DrawHandler.Callback
            public void danmakuShown(BaseDanmaku baseDanmaku) {
            }

            @Override // master.flame.danmaku.controller.DrawHandler.Callback
            public void drawingFinished() {
            }

            @Override // master.flame.danmaku.controller.DrawHandler.Callback
            public void prepared() {
                if (PlayModePlayActivity2.this.isPlay) {
                    PlayModePlayActivity2.this.mDanmakuView.start(PlayModePlayActivity2.this.mLineDrawView.getmPlayTime());
                }
            }

            @Override // master.flame.danmaku.controller.DrawHandler.Callback
            public void updateTimer(DanmakuTimer danmakuTimer) {
            }
        });
        this.mDanmakuView.showFPS(false);
        this.mDanmakuView.enableDanmakuDrawingCache(true);
    }

    private void initModePoint(boolean z) {
        String buildToUIData;
        String buildToUIData2;
        String str;
        String str2;
        PlayModeModel playModeModel = this.mPlayModeModel;
        if (playModeModel != null) {
            this.mTitleName.setText(playModeModel.getModeName());
            String modePoints = this.mPlayModeModel.getModePoints();
            String upperCase = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
            if (TextUtils.isEmpty(modePoints) || !modePoints.toUpperCase().contains(upperCase + Constans.START_TYPE_PIU_DATA1)) {
                this.img_mode_play_piu.setVisibility(8);
            } else {
                this.img_mode_play_piu.setVisibility(0);
            }
            String[] buildServerToClientData = LineDataUtil.buildServerToClientData(modePoints);
            LogUtil.e("modePointArr:" + Arrays.toString(buildServerToClientData));
            if (!TextUtils.isEmpty(buildServerToClientData[0])) {
                int parseInt = Integer.parseInt(buildServerToClientData[0]);
                if (buildServerToClientData.length == 5) {
                    this.mSMSModeType = 2;
                    buildToUIData = LineDataUtil.buildToUIData(buildServerToClientData[4], this.mLineDrawView.getIntervalTime(), parseInt);
                    str2 = LineDataUtil.buildToUIData(buildServerToClientData[1], this.mLineDrawView.getIntervalTime(), parseInt);
                    buildToUIData2 = LineDataUtil.buildToUIData(buildServerToClientData[2], this.mLineDrawView.getIntervalTime(), parseInt);
                    str = LineDataUtil.buildToUIData(buildServerToClientData[3], this.mLineDrawView.getIntervalTime(), parseInt);
                    this.layout_mark.setVisibility(0);
                } else {
                    this.mSMSModeType = 1;
                    buildToUIData = LineDataUtil.buildToUIData(buildServerToClientData[1], this.mLineDrawView.getIntervalTime(), parseInt);
                    LogUtil.e("modeData:" + buildToUIData);
                    buildToUIData2 = LineDataUtil.buildToUIData(buildServerToClientData[2], this.mLineDrawView.getIntervalTime(), parseInt);
                    LogUtil.e("modeVibrateData:" + buildToUIData2);
                    this.layout_mark.setVisibility(8);
                    str = "";
                    str2 = str;
                }
                boolean isEmpty = TextUtils.isEmpty(buildToUIData2);
                LineDrawView5 lineDrawView5 = this.mLineDrawView;
                if (isEmpty) {
                    lineDrawView5.setVibrateLine(false);
                } else {
                    lineDrawView5.setVibrateLine(true);
                }
                boolean isEmpty2 = TextUtils.isEmpty(str2);
                LineDrawView5 lineDrawView52 = this.mLineDrawView;
                if (isEmpty2) {
                    lineDrawView52.setStartPositionLine(false);
                } else {
                    lineDrawView52.setStartPositionLine(true);
                }
                boolean isEmpty3 = TextUtils.isEmpty(str);
                LineDrawView5 lineDrawView53 = this.mLineDrawView;
                if (isEmpty3) {
                    lineDrawView53.setSmoothLine(false);
                } else {
                    lineDrawView53.setSmoothLine(true);
                }
                this.mLineDrawView.setData(buildToUIData, buildToUIData2, str2, str);
            }
            if (!this.mLineDrawView.isDataEmpty()) {
                this.isPrepared = true;
            }
            this.mLineDrawView.setPlayTime(0);
            DanmakuView danmakuView = this.mDanmakuView;
            if (danmakuView != null && danmakuView.isPrepared()) {
                this.mDanmakuView.seekTo(0L);
            }
            this.mTxtPlayCount.setText(StringUtils.getFormatNum(this.mPlayModeModel.getPlayCount()));
            this.mTxtCollectCount.setText(StringUtils.getFormatNum(this.mPlayModeModel.getCollectCount()));
            this.mTxtLikeCount.setText(StringUtils.getFormatNum(this.mPlayModeModel.getLikeCount()));
            int liked = this.mPlayModeModel.getLiked();
            ImageView imageView = this.mImgLikeCount;
            if (liked == 1) {
                imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_mode_item_like_pressed));
            } else {
                imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_mode_item_like));
            }
            int collected = this.mPlayModeModel.getCollected();
            ImageView imageView2 = this.mImgCollectCount;
            if (collected == 1) {
                imageView2.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_mode_item_collect_pressed));
            } else {
                imageView2.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_mode_item_collect));
            }
            this.mTxtStartTime.setText(ModeTimeUtil.formatMilliTime(0));
            this.mSeekBar.setProgress(0);
            this.mTxtEndTime.setText(ModeTimeUtil.formatMilliTime(this.mLineDrawView.getmTime()));
            this.mSeekBar.setMax(this.mLineDrawView.getmTime());
            boolean isEmpty4 = TextUtils.isEmpty(this.mPlayModeModel.getModeSetting());
            ConstraintLayout constraintLayout = this.mLayoutSetting;
            if (isEmpty4) {
                constraintLayout.setVisibility(8);
            } else {
                constraintLayout.setVisibility(0);
                this.mPlayModeModel.initModeSettingsModel();
                LogUtil.e("重载模式 getModeSettingsModel:" + this.mPlayModeModel.getModeSettingsModel().toString());
                this.txt_setting_position_min.setText(this.mPlayModeModel.getModeSettingsModel().getStrokeMin() + "");
                this.txt_setting_position_max.setText(this.mPlayModeModel.getModeSettingsModel().getStrokeMax() + "");
                this.txt_setting_smooth.setText(this.mPlayModeModel.getModeSettingsModel().getSmooth() + "");
            }
            initProductData();
            if (z) {
                this.mHandler.sendEmptyMessageDelayed(1, this.delayMillis);
            }
            this.delayMillis = 0;
        }
    }

    private void initProductData() {
        if (BleManager.getInstance(this).getConnBleProduct(MyApplication.getApplication().getDeviceAddress()) != null) {
            BleDeviceModel connBleDevice = BleManager.getInstance(this).getConnBleProduct(MyApplication.getApplication().getDeviceAddress()).getConnBleDevice();
            if (ProductUtil.isSMSType()) {
                int i = this.mSMSModeType;
                if (i == 1) {
                    LogUtil.e("重载模式 runmode:" + connBleDevice.getRunMode());
                    if (connBleDevice.getRunMode() != 1) {
                        LogUtil.e("重载模式 connBleDevice.getRunMode():" + connBleDevice.getRunMode() + " 更新");
                        this.mHandler.sendEmptyMessageDelayed(2, this.delayMillis);
                        this.delayMillis += this.defDelayMillils;
                    }
                    PlayModeModel playModeModel = this.mPlayModeModel;
                    if (playModeModel != null && playModeModel.getModeSettingsModel() != null) {
                        LogUtil.e("重载模式 getStartPosition:" + this.mPlayModeModel.getModeSettingsModel().getStrokeMin());
                        if (connBleDevice.getStartPosition() != this.mPlayModeModel.getModeSettingsModel().getStrokeMin()) {
                            LogUtil.e("重载模式 connBleDevice.getStartPosition():" + connBleDevice.getStartPosition() + " 更新");
                            this.mHandler.sendEmptyMessageDelayed(3, this.delayMillis);
                            this.delayMillis += this.defDelayMillils;
                        }
                        LogUtil.e("重载模式 getStrokeMax:" + this.mPlayModeModel.getModeSettingsModel().getStrokeMax());
                        if (connBleDevice.getStopPosition() != this.mPlayModeModel.getModeSettingsModel().getStrokeMax()) {
                            LogUtil.e("重载模式 connBleDevice.getStopPosition():" + connBleDevice.getStopPosition() + " 更新");
                            this.mHandler.sendEmptyMessageDelayed(4, this.delayMillis);
                            this.delayMillis += this.defDelayMillils;
                        }
                        LogUtil.e("重载模式 getSmooth:" + this.mPlayModeModel.getModeSettingsModel().getSmooth());
                        if (connBleDevice.getSmoothness() != this.mPlayModeModel.getModeSettingsModel().getSmooth()) {
                            LogUtil.e("重载模式 connBleDevice.getSmoothness():" + connBleDevice.getSmoothness() + " 更新");
                            this.mHandler.sendEmptyMessageDelayed(5, this.delayMillis);
                            this.delayMillis += this.defDelayMillils;
                        }
                    }
                } else if (i == 2) {
                    LogUtil.e("重载模式 runmode:" + connBleDevice.getRunMode());
                    if (connBleDevice.getRunMode() != 0) {
                        LogUtil.e("重载模式 connBleDevice.getRunMode():" + connBleDevice.getRunMode() + " 更新");
                        this.mHandler.sendEmptyMessageDelayed(6, this.delayMillis);
                        this.delayMillis += this.defDelayMillils;
                    }
                }
            }
        }
        if (BleManager.getInstance(this).hasPiuFunBle()) {
            plugPiuTimerFragment();
        } else {
            removePlugPiuTimerFragment();
        }
    }

    private void initRecyclerColor() {
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this, 8);
        this.mLayoutManager = gridLayoutManager;
        this.mRecyclerColor.setLayoutManager(gridLayoutManager);
        PlayModeColorAdapter playModeColorAdapter = new PlayModeColorAdapter(this);
        this.mAdapter = playModeColorAdapter;
        playModeColorAdapter.setOnItemClickListener(new PlayModeColorAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.2
            @Override // com.hismith.hismithapp.adapter.PlayModeColorAdapter.OnItemClickListener
            public void onItemClick(String str) {
                PlayModePlayActivity2.this.mViewBarrageColor.setBackgroundColor(Color.parseColor(str));
                PlayModePlayActivity2.this.mRecyclerColor.setVisibility(4);
            }
        });
        this.mRecyclerColor.setAdapter(this.mAdapter);
    }

    private void initSensor() {
        SensorManager sensorManager = (SensorManager) getSystemService("sensor");
        this.mSensorManager = sensorManager;
        Sensor defaultSensor = sensorManager.getDefaultSensor(1);
        this.mSensor = defaultSensor;
        this.mSensorManager.registerListener(this, defaultSensor, 2);
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.img_auto_play = (ImageView) findViewById(R.id.img_auto_play);
        this.img_mode_play_piu = (ImageView) findViewById(R.id.img_mode_play_piu);
        this.mLineDrawView = (LineDrawView5) findViewById(R.id.mode_play_lineDrawView);
        this.mDanmakuView = (DanmakuView) findViewById(R.id.mode_play_danmaku);
        this.mImgPlayCount = (ImageView) findViewById(R.id.mode_play_img_playCount);
        this.mImgCollectCount = (ImageView) findViewById(R.id.mode_play_img_collectCount);
        this.mImgLikeCount = (ImageView) findViewById(R.id.mode_play_img_likeCount);
        this.mImgMore = (ImageView) findViewById(R.id.mode_play_img_more);
        this.mImgPlay = (ImageView) findViewById(R.id.mode_play_img_play);
        this.mImgType = (ImageView) findViewById(R.id.mode_play_img_type);
        this.mTxtPlayCount = (TextView) findViewById(R.id.mode_play_txt_playCount);
        this.mTxtCollectCount = (TextView) findViewById(R.id.mode_play_txt_collectCount);
        this.mTxtLikeCount = (TextView) findViewById(R.id.mode_play_txt_likeCount);
        this.mTxtStartTime = (TextView) findViewById(R.id.mode_play_txt_start_time);
        this.mTxtEndTime = (TextView) findViewById(R.id.mode_play_txt_end_time);
        this.mSeekBar = (SeekBar) findViewById(R.id.mode_play_seek_bar);
        this.mLayoutBarrage = (ConstraintLayout) findViewById(R.id.mode_play_layout_barrage);
        this.mImgBarrage = (ImageView) findViewById(R.id.mode_play_img_barrage);
        this.mLayoutBarrageColor = (FrameLayout) findViewById(R.id.mode_play_layout_barrage_color);
        this.mViewBarrageColor = findViewById(R.id.mode_play_view_barrage_color);
        this.mEditBarrageContent = (EditText) findViewById(R.id.mode_play_edit_barrage_content);
        this.mImgBarrageSend = (ImageView) findViewById(R.id.mode_play_img_barrage_send);
        this.mRecyclerColor = (RecyclerView) findViewById(R.id.mode_play_recycler_color);
        RightDraggableFrameLayout rightDraggableFrameLayout = (RightDraggableFrameLayout) findViewById(R.id.layout_fragment_piu);
        this.layout_fragment_piu = rightDraggableFrameLayout;
        rightDraggableFrameLayout.setTopBoundViewId(R.id.mode_play_layout_action);
        this.layout_fragment_piu.setBottomBoundViewId(R.id.mode_play_layout_barrage_f);
        this.layout_mark = (LinearLayout) findViewById(R.id.layout_mark);
        this.txt_mode_speed = (TextView) findViewById(R.id.txt_mode_speed);
        this.txt_mode_max_stroke = (TextView) findViewById(R.id.txt_mode_max_stroke);
        this.txt_mode_depth = (TextView) findViewById(R.id.txt_mode_depth);
        this.txt_mode_smooth = (TextView) findViewById(R.id.txt_mode_smooth);
        LanguageUtil.setText(this, this.txt_mode_speed, "speed");
        LanguageUtil.setText(this, this.txt_mode_max_stroke, "maxStroke");
        LanguageUtil.setText(this, this.txt_mode_depth, "Depth");
        LanguageUtil.setText(this, this.txt_mode_smooth, "smoothness");
        this.mLayoutSetting = (ConstraintLayout) findViewById(R.id.layout_setting);
        this.txt_setting_position_min = (TextView) findViewById(R.id.txt_setting_position_min);
        this.txt_setting_position_max = (TextView) findViewById(R.id.txt_setting_position_max);
        this.txt_setting_smooth = (TextView) findViewById(R.id.txt_setting_smooth);
        this.mLineDrawView.setPlayCircle(true);
        this.mLineDrawView.setDrawYScale(false);
        this.mLineDrawView.setPiuDragRemoveEnabled(false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void pauseMode() {
        this.isPlay = false;
        this.mImgPlay.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_mode_play));
        this.mLineDrawView.pause();
        DanmakuView danmakuView = this.mDanmakuView;
        if (danmakuView == null || !danmakuView.isPrepared()) {
            return;
        }
        this.mDanmakuView.pause();
    }

    private void playMode() {
        this.isPlay = true;
        this.mImgPlay.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_mode_pause));
        boolean z = this.isBarrageShow;
        DanmakuView danmakuView = this.mDanmakuView;
        if (z) {
            if (danmakuView != null && danmakuView.isPrepared()) {
                boolean isShown = this.mDanmakuView.isShown();
                DanmakuView danmakuView2 = this.mDanmakuView;
                if (!isShown) {
                    danmakuView2.showAndResumeDrawTask(Long.valueOf(this.mLineDrawView.getmPlayTime()));
                } else if (danmakuView2 != null && danmakuView2.isPrepared()) {
                    this.mDanmakuView.start(this.mLineDrawView.getmPlayTime());
                }
            }
        } else if (danmakuView != null && danmakuView.isPrepared()) {
            this.mDanmakuView.start(this.mLineDrawView.getmPlayTime());
        }
        this.mLineDrawView.play();
        if (BleManager.getInstance(this).getConnBleProduct(MyApplication.getApplication().getDeviceAddress()) != null) {
            BleDeviceModel connBleDevice = BleManager.getInstance(this).getConnBleProduct(MyApplication.getApplication().getDeviceAddress()).getConnBleDevice();
            if (ProductUtil.isSMSType()) {
                int i = this.mSMSModeType;
                if (i == 1) {
                    if (connBleDevice.getRunMode() != 1) {
                        LogUtil.e("重载模式 connBleDevice.getRunMode():" + connBleDevice.getRunMode() + " 更新");
                        this.mHandler.sendEmptyMessageDelayed(2, this.delayMillis);
                        return;
                    }
                    return;
                }
                if (i != 2 || connBleDevice.getRunMode() == 0) {
                    return;
                }
                LogUtil.e("重载模式 connBleDevice.getRunMode():" + connBleDevice.getRunMode() + " 更新");
                this.mHandler.sendEmptyMessageDelayed(6, this.delayMillis);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeAction(final String str, final String str2, boolean z) {
        if (this.mPlayModeModel == null) {
            return;
        }
        this.baseRequest.getApi().playModeAction((String) SharedPreferencesUtil.get(this, Constans.Token), this.mPlayModeModel.getModeId(), str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeActionModel>(MyApplication.getApplication(), this.progressDialog, z) { // from class: com.hismith.hismithapp.PlayModePlayActivity2.13
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeActionModel playModeActionModel) {
                if (playModeActionModel.getSuccess() == 1) {
                    if ("like".equals(str)) {
                        PlayModePlayActivity2.this.mPlayModeModel.setLiked(1);
                        PlayModePlayActivity2.this.mPlayModeModel.setLikeCount(playModeActionModel.getLikeCount());
                        PlayModePlayActivity2.this.mTxtLikeCount.setText(StringUtils.getFormatNum(playModeActionModel.getLikeCount()));
                        PlayModePlayActivity2.this.mImgLikeCount.setImageResource(R.mipmap.ic_play_mode_item_like_pressed);
                        return;
                    }
                    if ("unlike".equals(str)) {
                        PlayModePlayActivity2.this.mPlayModeModel.setLiked(0);
                        PlayModePlayActivity2.this.mPlayModeModel.setLikeCount(playModeActionModel.getLikeCount());
                        PlayModePlayActivity2.this.mTxtLikeCount.setText(StringUtils.getFormatNum(playModeActionModel.getLikeCount()));
                        PlayModePlayActivity2.this.mImgLikeCount.setImageResource(R.mipmap.ic_play_mode_item_like);
                        return;
                    }
                    if ("collect".equals(str)) {
                        PlayModePlayActivity2.this.mPlayModeModel.setCollected(1);
                        PlayModePlayActivity2.this.mPlayModeModel.setCollectCount(playModeActionModel.getCollectCount());
                        PlayModePlayActivity2.this.mTxtCollectCount.setText(StringUtils.getFormatNum(playModeActionModel.getCollectCount()));
                        PlayModePlayActivity2.this.mImgCollectCount.setImageResource(R.mipmap.ic_play_mode_item_collect_pressed);
                        return;
                    }
                    if ("uncollect".equals(str)) {
                        PlayModePlayActivity2.this.mPlayModeModel.setCollected(0);
                        PlayModePlayActivity2.this.mPlayModeModel.setCollectCount(playModeActionModel.getCollectCount());
                        PlayModePlayActivity2.this.mTxtCollectCount.setText(StringUtils.getFormatNum(playModeActionModel.getCollectCount()));
                        PlayModePlayActivity2.this.mImgCollectCount.setImageResource(R.mipmap.ic_play_mode_item_collect);
                        return;
                    }
                    if ("report".equals(str)) {
                        ToastUtil.showShortToastCenter(PlayModePlayActivity2.this, LanguageUtil.getLanguageByKey(PlayModePlayActivity2.this, "reportedSuccess"));
                        return;
                    }
                    if ("rename".equals(str)) {
                        PlayModePlayActivity2.this.mPlayModeModel.setModeName(str2);
                        PlayModePlayActivity2.this.mTitleName.setText(PlayModePlayActivity2.this.mPlayModeModel.getModeName());
                    } else if ("share".equals(str)) {
                        ToastUtil.showShortToastCenter(PlayModePlayActivity2.this, LanguageUtil.getLanguageByKey(PlayModePlayActivity2.this, "complete"));
                    } else if ("del".equals(str)) {
                        ToastUtil.showShortToastCenter(PlayModePlayActivity2.this, LanguageUtil.getLanguageByKey(PlayModePlayActivity2.this, "complete"));
                        PlayModePlayActivity2.this.finish();
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeAction(String str, boolean z) {
        playModeAction(str, null, z);
    }

    private void plugPiuTimerFragment() {
        FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
        this.mPlugPiuTimer = new PlugPiuTimer();
        beginTransaction.replace(R.id.layout_fragment_piu, this.mPlugPiuTimer);
        beginTransaction.commitAllowingStateLoss();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void prepareDanmu(BarrageModel barrageModel) {
        BaseDanmakuParser createParser = createParser(barrageModel);
        this.mParser = createParser;
        this.mDanmakuView.prepare(createParser, this.danmakuContext);
    }

    private void removePlugPiuTimerFragment() {
        if (this.mPlugPiuTimer != null) {
            FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
            beginTransaction.remove(this.mPlugPiuTimer);
            beginTransaction.commitAllowingStateLoss();
            this.mPlugPiuTimer = null;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendBarrage() {
        String trim = this.mEditBarrageContent.getText().toString().trim();
        if (TextUtils.isEmpty(trim)) {
            return;
        }
        setPlayModeBarrage(trim, this.mSeekBar.getProgress() / 1000.0f, getSelectedColor());
        this.mEditBarrageContent.setText("");
    }

    private void setBarrageShow() {
        boolean z = this.isBarrageShow;
        DanmakuView danmakuView = this.mDanmakuView;
        if (z) {
            if (danmakuView != null && danmakuView.isPrepared()) {
                this.mDanmakuView.show();
            }
            this.mImgBarrage.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_open));
            this.mLayoutBarrageColor.setVisibility(0);
            this.mEditBarrageContent.setVisibility(0);
            this.mImgBarrageSend.setVisibility(0);
            return;
        }
        if (danmakuView != null && danmakuView.isPrepared()) {
            this.mDanmakuView.hide();
        }
        this.mImgBarrage.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_close));
        this.mLayoutBarrageColor.setVisibility(4);
        this.mEditBarrageContent.setVisibility(4);
        this.mImgBarrageSend.setVisibility(4);
        this.mRecyclerColor.setVisibility(4);
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.img_auto_play.setOnClickListener(this);
        this.img_mode_play_piu.setOnClickListener(this);
        this.mLineDrawView.setOnPlayTimeListener(this);
        this.mImgCollectCount.setOnClickListener(this);
        this.mImgLikeCount.setOnClickListener(this);
        this.mImgMore.setOnClickListener(this);
        this.mImgPlay.setOnClickListener(this);
        this.mImgType.setOnClickListener(this);
        this.mSeekBar.setOnSeekBarChangeListener(this);
        this.mImgBarrage.setOnClickListener(this);
        this.mLayoutBarrageColor.setOnClickListener(this);
        this.mImgBarrageSend.setOnClickListener(this);
        this.mEditBarrageContent.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.4
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4) {
                    return false;
                }
                PlayModePlayActivity2.this.sendBarrage();
                KeyboardUtils.hideSoftInput(textView);
                return true;
            }
        });
        this.mTxtStartTime.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.5
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlayModePlayActivity2.this.mTxtStartTime.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                EditText editText = PlayModePlayActivity2.this.mEditBarrageContent;
                PlayModePlayActivity2 playModePlayActivity2 = PlayModePlayActivity2.this;
                editText.setTextSize(playModePlayActivity2.autosizeText(playModePlayActivity2.mTxtStartTime.getTextSize()));
            }
        });
    }

    private void setPlayModeBarrage(final String str, double d, final String str2) {
        if (this.mPlayModeModel == null) {
            return;
        }
        this.baseRequest.getApi().setPlayModeBarrage((String) SharedPreferencesUtil.get(this, Constans.Token), this.mPlayModeModel.getModeId(), d, str2, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<BarrageModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.PlayModePlayActivity2.16
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(BarrageModel barrageModel) {
                if (barrageModel.getSuccess() == 1) {
                    PlayModePlayActivity2.this.addDanmaku(str, str2);
                }
            }
        });
    }

    private void showMorePopup(View view) {
        ActionAttachPopup2 actionAttachPopup2 = new ActionAttachPopup2(this, getActionList());
        actionAttachPopup2.setCollectClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                int collected = PlayModePlayActivity2.this.mPlayModeModel.getCollected();
                PlayModePlayActivity2 playModePlayActivity2 = PlayModePlayActivity2.this;
                if (collected == 1) {
                    playModePlayActivity2.playModeAction("uncollect", true);
                } else {
                    playModePlayActivity2.playModeAction("collect", true);
                }
            }
        });
        actionAttachPopup2.setOnReportClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.7
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                new ReportSelectDialog(PlayModePlayActivity2.this) { // from class: com.hismith.hismithapp.PlayModePlayActivity2.7.1
                    @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                    public void btnSubmit(String str, String str2) {
                        PlayModePlayActivity2.this.complaint(PlayModePlayActivity2.this.mPlayModeModel.getModeId(), str, str2);
                    }
                }.showDialog();
            }
        });
        actionAttachPopup2.setOnShareClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.8
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                DialogUtil.showConfirmDialog(PlayModePlayActivity2.this, "prompt", "shareMode", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.8.1
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view3) {
                        PlayModePlayActivity2.this.playModeAction("share", true);
                        DialogUtil.dismissDialog();
                    }
                });
            }
        });
        actionAttachPopup2.setOnRenameClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.9
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                PlayModePlayActivity2 playModePlayActivity2 = PlayModePlayActivity2.this;
                new RenameDialog(playModePlayActivity2, playModePlayActivity2.mPlayModeModel.getModeName()) { // from class: com.hismith.hismithapp.PlayModePlayActivity2.9.1
                    @Override // com.hismith.hismithapp.dialog.RenameDialog
                    public void btnSubmit(String str) {
                        PlayModePlayActivity2.this.playModeAction("rename", str, true);
                    }
                }.show();
            }
        });
        actionAttachPopup2.setOnDelClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.10
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                DialogUtil.showConfirmDialog(PlayModePlayActivity2.this, "prompt", "deleteMode", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.10.1
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view3) {
                        PlayModePlayActivity2.this.playModeAction("del", true);
                        DialogUtil.dismissDialog();
                    }
                });
            }
        });
        actionAttachPopup2.setOnEditClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.11
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                int status = BleManager.getInstance(PlayModePlayActivity2.this).getStatus();
                PlayModePlayActivity2 playModePlayActivity2 = PlayModePlayActivity2.this;
                if (status != 2) {
                    ToastUtil.showShortToastCenter(PlayModePlayActivity2.this, LanguageUtil.getLanguageByKey(playModePlayActivity2, "noConnectedDevice"));
                    return;
                }
                playModePlayActivity2.pauseMode();
                if (TextUtils.isEmpty(PlayModePlayActivity2.this.mPlayModeModel.getModeData())) {
                    Intent intent = new Intent(PlayModePlayActivity2.this, (Class<?>) PlayModeCreateActivity2.class);
                    intent.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "edit");
                    intent.putExtra("editMode", PlayModePlayActivity2.this.mPlayModeModel);
                    PlayModePlayActivity2.this.startActivityForResult(intent, 18);
                    return;
                }
                Intent intent2 = new Intent(PlayModePlayActivity2.this, (Class<?>) PlayModeCreateActivity.class);
                intent2.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "edit");
                intent2.putExtra("editMode", PlayModePlayActivity2.this.mPlayModeModel);
                PlayModePlayActivity2.this.startActivityForResult(intent2, 18);
            }
        });
        actionAttachPopup2.setOnCopyClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.12
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                int status = BleManager.getInstance(PlayModePlayActivity2.this).getStatus();
                PlayModePlayActivity2 playModePlayActivity2 = PlayModePlayActivity2.this;
                if (status != 2) {
                    ToastUtil.showShortToastCenter(PlayModePlayActivity2.this, LanguageUtil.getLanguageByKey(playModePlayActivity2, "noConnectedDevice"));
                    return;
                }
                playModePlayActivity2.pauseMode();
                if (TextUtils.isEmpty(PlayModePlayActivity2.this.mPlayModeModel.getModeData())) {
                    Intent intent = new Intent(PlayModePlayActivity2.this, (Class<?>) PlayModeCreateActivity2.class);
                    intent.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "copy");
                    intent.putExtra("editMode", PlayModePlayActivity2.this.mPlayModeModel);
                    PlayModePlayActivity2.this.startActivityForResult(intent, 18);
                    return;
                }
                Intent intent2 = new Intent(PlayModePlayActivity2.this, (Class<?>) PlayModeCreateActivity.class);
                intent2.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "copy");
                intent2.putExtra("editMode", PlayModePlayActivity2.this.mPlayModeModel);
                PlayModePlayActivity2.this.startActivityForResult(intent2, 18);
            }
        });
        actionAttachPopup2.showPop(view, MeasureUtil.dip2px(this, 8.0f));
    }

    @Override // android.app.Activity
    public void finish() {
        if (this.mModeChanged) {
            setResult(17, new Intent());
        }
        super.finish();
    }

    @Override // android.hardware.SensorEventListener
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        LogUtil.e("创建模式界面返回requestCode：" + i + ", resultCode:" + i2);
        if (i != 18 || i2 != 17 || intent == null || this.mPlayModeModel == null) {
            return;
        }
        String stringExtra = intent.getStringExtra("modeData");
        String stringExtra2 = intent.getStringExtra("modePoint");
        String stringExtra3 = intent.getStringExtra("modeSetting");
        this.mPlayModeModel.setModeData(stringExtra);
        this.mPlayModeModel.setModePoints(stringExtra2);
        this.mPlayModeModel.setModeSetting(stringExtra3);
        initModePoint(false);
        this.mModeChanged = true;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.img_auto_play) {
            boolean z = this.isAutoPlay;
            this.isAutoPlay = !z;
            ImageView imageView = this.img_auto_play;
            if (z) {
                imageView.setImageResource(R.mipmap.ic_auto_switch_off);
            } else {
                imageView.setImageResource(R.mipmap.ic_auto_switch_on);
            }
            SharedPreferencesUtil.put(this, Constans.ModePlayAutoPlay, Boolean.valueOf(this.isAutoPlay));
            return;
        }
        if (id == R.id.img_mode_play_piu) {
            boolean z2 = this.isShowPiuUI;
            this.isShowPiuUI = !z2;
            ImageView imageView2 = this.img_mode_play_piu;
            if (z2) {
                imageView2.setImageResource(R.mipmap.ic_piu_switch_off);
            } else {
                imageView2.setImageResource(R.mipmap.ic_piu_switch_on);
            }
            this.mLineDrawView.setDrawPiu(this.isShowPiuUI);
            return;
        }
        if (id == R.id.mode_play_img_collectCount) {
            PlayModeModel playModeModel = this.mPlayModeModel;
            if (playModeModel != null) {
                if (playModeModel.getCollected() == 1) {
                    playModeAction("uncollect", true);
                    return;
                } else {
                    playModeAction("collect", true);
                    return;
                }
            }
            return;
        }
        if (id == R.id.mode_play_img_likeCount) {
            PlayModeModel playModeModel2 = this.mPlayModeModel;
            if (playModeModel2 != null) {
                if (playModeModel2.getLiked() == 1) {
                    playModeAction("unlike", true);
                    return;
                } else {
                    playModeAction("like", true);
                    return;
                }
            }
            return;
        }
        if (id == R.id.mode_play_img_play) {
            if (this.isPrepared) {
                if (this.isPlay) {
                    pauseMode();
                    return;
                } else {
                    playMode();
                    return;
                }
            }
            return;
        }
        if (id == R.id.mode_play_img_type) {
            if (this.mPlayType == 1) {
                this.mPlayType = 0;
                this.mImgType.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_type_single));
                ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "singlePlay"));
                return;
            } else {
                this.mPlayType = 1;
                this.mImgType.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_type_cycle));
                ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "loopPlay"));
                return;
            }
        }
        if (id == R.id.mode_play_img_more) {
            if (this.mPlayModeModel != null) {
                showMorePopup(view);
                return;
            }
            return;
        }
        if (id == R.id.mode_play_img_barrage) {
            this.isBarrageShow = !this.isBarrageShow;
            setBarrageShow();
            return;
        }
        if (id == R.id.mode_play_img_barrage_send) {
            sendBarrage();
            KeyboardUtils.hideSoftInput(view);
        } else if (id == R.id.mode_play_layout_barrage_color) {
            int visibility = this.mRecyclerColor.getVisibility();
            RecyclerView recyclerView = this.mRecyclerColor;
            if (visibility == 0) {
                recyclerView.setVisibility(4);
            } else {
                recyclerView.setVisibility(0);
                KeyboardUtils.hideSoftInput(view);
            }
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onComplete() {
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.17
            @Override // java.lang.Runnable
            public void run() {
                int i = PlayModePlayActivity2.this.mPlayType;
                PlayModePlayActivity2 playModePlayActivity2 = PlayModePlayActivity2.this;
                if (i == 1) {
                    if (playModePlayActivity2.mLineDrawView != null) {
                        PlayModePlayActivity2.this.mLineDrawView.play();
                    }
                    if (PlayModePlayActivity2.this.mDanmakuView != null && PlayModePlayActivity2.this.mDanmakuView.isPrepared()) {
                        PlayModePlayActivity2.this.mDanmakuView.seekTo(Long.valueOf(PlayModePlayActivity2.this.mLineDrawView.getmPlayTime()));
                    }
                } else {
                    playModePlayActivity2.isPlay = false;
                    PlayModePlayActivity2.this.lastSpeed = 0;
                    PlayModePlayActivity2.this.lastVibrate = 0;
                    if (PlayModePlayActivity2.this.mLineDrawView != null) {
                        PlayModePlayActivity2.this.mLineDrawView.pause();
                    }
                    PlayModePlayActivity2.this.mImgPlay.setImageDrawable(ContextCompat.getDrawable(PlayModePlayActivity2.this, R.mipmap.ic_mode_play));
                    if (PlayModePlayActivity2.this.mDanmakuView != null && PlayModePlayActivity2.this.mDanmakuView.isPrepared()) {
                        PlayModePlayActivity2.this.mDanmakuView.pause();
                    }
                }
                if (BleManager.getInstance(PlayModePlayActivity2.this).getmConnBleDevice().getReversal() == 1) {
                    BleManager.getInstance(PlayModePlayActivity2.this).writeCloseReversal();
                }
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getWindow().addFlags(128);
        setContentView(R.layout.activity_play_mode_play2);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        DanmakuView danmakuView = this.mDanmakuView;
        if (danmakuView != null) {
            danmakuView.release();
            this.mDanmakuView = null;
        }
        this.mLineDrawView.pause();
        PiuTimerManager.getInstance(this).pauseTimer();
        this.mLineDrawView.setOnPlayTimeListener(null);
        BleManager.getInstance(this).writeShutdown();
        BleManager.getInstance(this).setOnConnectStateListener(null);
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onFunData(String str, String str2) {
        int intValue = Integer.valueOf(str, 16).intValue();
        int intValue2 = Integer.valueOf(str2, 16).intValue();
        if (intValue == 254) {
            if (intValue2 == 1) {
                BleManager.getInstance(this).writeOpenPiu();
            }
        } else {
            if (intValue != 255) {
                return;
            }
            if (intValue2 == 0) {
                BleManager.getInstance(this).writeCloseReversal();
            } else if (intValue2 == 1) {
                BleManager.getInstance(this).writeOpenReversal();
            }
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onLinePause() {
        LogUtil.e("onLinePause()");
        if (ProductUtil.isSMSType() && this.mSMSModeType == 1) {
            return;
        }
        BleManager.getInstance(this).writeSpeed(0);
        this.lastSpeed = 0;
        if (!this.mLineDrawView.isVibrateLine() || ProductUtil.isSMSType()) {
            return;
        }
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.21
            @Override // java.lang.Runnable
            public void run() {
                BleManager.getInstance(PlayModePlayActivity2.this).writeModeLine2(0);
                PlayModePlayActivity2.this.lastVibrate = 0;
            }
        }, 100L);
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onPlayTime(int i) {
        if (this.isPrepared && !this.isSendPlay && i > this.mSeekBar.getMax() / 2) {
            this.isSendPlay = true;
            playModeAction("play", false);
        }
        this.mTxtStartTime.setText(ModeTimeUtil.formatMilliTime(i));
        if (this.isDownSeek) {
            return;
        }
        this.mSeekBar.setProgress(i);
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onPositionRangeChange(final int i, final int i2) {
        int i3 = 15;
        int i4 = 30;
        if (i >= this.lastStopPosition) {
            i4 = 15;
            i3 = 30;
        }
        if (i != this.lastStartPosition) {
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.18
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(PlayModePlayActivity2.this).writeData(BleManager.getInstance(PlayModePlayActivity2.this).getBleCCData(7, i));
                    PlayModePlayActivity2.this.lastStartPosition = i;
                }
            }, i3);
        }
        if (i2 != this.lastStopPosition) {
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.19
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(PlayModePlayActivity2.this).writeData(BleManager.getInstance(PlayModePlayActivity2.this).getBleCCData(8, i2));
                    PlayModePlayActivity2.this.lastStopPosition = i2;
                }
            }, i4);
        }
    }

    @Override // android.widget.SeekBar.OnSeekBarChangeListener
    public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
    }

    @Override // android.hardware.SensorEventListener
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == 1) {
            float f = sensorEvent.values[0];
            float f2 = sensorEvent.values[1];
            float f3 = sensorEvent.values[2];
            if (f3 <= -9.0f && f3 > -10.0f) {
                int i = this.reversalCount + 1;
                this.reversalCount = i;
                if (i >= 4) {
                    this.recoverCount = 0;
                    if (this.isReversal) {
                        return;
                    }
                    LogUtil.e("Sensor value 翻转翻转翻转");
                    this.isReversal = true;
                    if (this.isPrepared && this.isPlay) {
                        pauseMode();
                        return;
                    }
                    return;
                }
                return;
            }
            if (f3 > -1.0f) {
                int i2 = this.recoverCount + 1;
                this.recoverCount = i2;
                if (i2 >= 4) {
                    this.reversalCount = 0;
                    if (this.isReversal) {
                        LogUtil.e("Sensor value 翻转恢复");
                        this.isReversal = false;
                        if (!this.isPrepared || this.isPlay) {
                            return;
                        }
                        playMode();
                    }
                }
            }
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onSmoothChange(final int i) {
        if (i != this.lastSmooth) {
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayModePlayActivity2.20
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(PlayModePlayActivity2.this).writeData(BleManager.getInstance(PlayModePlayActivity2.this).getBleCCData(9, i));
                    PlayModePlayActivity2.this.lastSmooth = i;
                }
            }, 45L);
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onSpeedChange(int i) {
        if (i != this.lastSpeed) {
            LogUtil.e("发送：" + i);
            if (this.mSMSModeType == 1 && ProductUtil.isSMSType()) {
                BleManager.getInstance(this).writePosition(i);
            } else {
                BleManager.getInstance(this).writeSpeed(i);
            }
            this.lastSpeed = i;
        }
    }

    @Override // android.widget.SeekBar.OnSeekBarChangeListener
    public void onStartTrackingTouch(SeekBar seekBar) {
        this.isDownSeek = true;
    }

    @Override // android.widget.SeekBar.OnSeekBarChangeListener
    public void onStopTrackingTouch(SeekBar seekBar) {
        this.isDownSeek = false;
        int progress = seekBar.getProgress();
        int i = progress % 5;
        if (i != 0) {
            progress -= i;
        }
        this.mLineDrawView.setPlayTime(progress);
        boolean isPlay = this.mLineDrawView.isPlay();
        DanmakuView danmakuView = this.mDanmakuView;
        if (isPlay) {
            if (danmakuView == null || !danmakuView.isPrepared()) {
                return;
            }
            this.mDanmakuView.seekTo(Long.valueOf(progress));
            return;
        }
        if (danmakuView == null || !danmakuView.isPrepared()) {
            return;
        }
        this.mDanmakuView.hide();
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onVibrateChange(int i) {
        if (i != this.lastVibrate) {
            BleManager.getInstance(this).writeModeLine2(i);
            this.lastVibrate = i;
        }
    }
}
