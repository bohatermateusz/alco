package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ModeTimeUtil;
import com.hismith.hismithapp.view.LineChartPlayView3;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class PreviewDialog extends Dialog implements View.OnClickListener, LineChartPlayView3.OnPlayListener, SeekBar.OnSeekBarChangeListener {
    private final int delay;
    private Map<Integer, String> funDataMap;
    private boolean isDownSeek;
    private boolean isPrepared;
    private int lastStart;
    private int lastStop;
    private LinearLayout layout_mark;
    private Activity mActivity;
    private Handler mHandler;
    private ImageView mImgClose;
    private LineChartPlayView3 mLineChartPlay;
    private String mModeData;
    private int mSMSModeType;
    private SeekBar mSeekBar;
    private TextView mTxtEndTime;
    private TextView mTxtStartTime;
    int timeSecond;
    private TextView txt_mode_depth;
    private TextView txt_mode_max_stroke;
    private TextView txt_mode_smooth;
    private TextView txt_mode_speed;

    public PreviewDialog(Activity activity, String str) {
        super(activity);
        this.isPrepared = false;
        this.isDownSeek = false;
        this.mSMSModeType = 1;
        this.lastStart = 0;
        this.lastStop = 100;
        this.mHandler = new Handler(Looper.getMainLooper());
        this.timeSecond = 0;
        this.delay = 15;
        this.mActivity = activity;
        this.mModeData = str;
        if (!TextUtils.isEmpty(str)) {
            String substring = this.mModeData.substring(0, 1);
            if ("1".equals(substring)) {
                this.mSMSModeType = 1;
            } else if ("S".equals(substring)) {
                this.mSMSModeType = 2;
            }
        }
        this.funDataMap = new HashMap();
    }

    private void initLoopData(List<String> list, int i, int i2, int i3) {
        if (list.size() < i || i <= 0 || list.size() < i2 || i2 <= 0) {
            return;
        }
        for (int i4 = 0; i4 < i3; i4++) {
            for (int i5 = i; i5 <= i2; i5++) {
                initSingleData(list.get(i5 - 1));
            }
        }
    }

    private void initModeData(String[] strArr) {
        ArrayList arrayList = new ArrayList();
        for (int i = 0; i < strArr.length; i++) {
            if (strArr[i].length() > 5) {
                if ("0".equals(strArr[i].substring(0, 1))) {
                    initLoopData(arrayList, Integer.valueOf(strArr[i].substring(1, 3), 16).intValue(), Integer.valueOf(strArr[i].substring(3, 5), 16).intValue(), Integer.valueOf(strArr[i].substring(5, 7), 16).intValue());
                } else {
                    arrayList.add(strArr[i]);
                    initSingleData(strArr[i]);
                }
            }
        }
    }

    private void initPoints(String[] strArr, int i) {
    }

    private void initProductData() {
        if (ProductUtil.isSMSType() && this.mSMSModeType == 1) {
            BleManager.getInstance(this.mActivity).writeRunManual();
        }
    }

    private void initSingleData(String str) {
        if (str.length() > 5) {
            if (Integer.valueOf(str.substring(1, 3), 16).intValue() == 255) {
                this.funDataMap.put(Integer.valueOf(this.timeSecond), str.substring(1, 5));
                return;
            }
            String substring = str.substring(0, 1);
            if ("1".equals(substring) && str.length() >= 8) {
                this.timeSecond += (Integer.valueOf(str.substring(5, 7), 16).intValue() * 100) + (Integer.valueOf(str.substring(7, 8), 16).intValue() * 10);
                return;
            }
            if ("2".equals(substring) && str.length() >= 12) {
                this.timeSecond += (Integer.valueOf(str.substring(9, 11), 16).intValue() * 100) + (Integer.valueOf(str.substring(11, 12), 16).intValue() * 10);
            } else {
                if (!"S".equals(substring) || str.length() < 13) {
                    return;
                }
                this.timeSecond += (Integer.valueOf(str.substring(10, 12), 16).intValue() * 100) + (Integer.valueOf(str.substring(12, 13), 16).intValue() * 10);
            }
        }
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        if (this.isPrepared) {
            initProductData();
            this.mLineChartPlay.startSync();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        dismiss();
    }

    @Override // com.hismith.hismithapp.view.LineChartPlayView3.OnPlayListener
    public void onComplete() {
        if (ProductUtil.isSMSType() && this.mSMSModeType == 1) {
            return;
        }
        BleManager.getInstance(this.mActivity).writeShutdown();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_preview);
        Window window = getWindow();
        if (window != null) {
            window.setBackgroundDrawable(null);
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (this.mActivity.getWindowManager().getDefaultDisplay().getWidth() * 0.9d);
            window.setAttributes(attributes);
        }
        this.layout_mark = (LinearLayout) findViewById(R.id.layout_mark);
        this.txt_mode_speed = (TextView) findViewById(R.id.txt_mode_speed);
        this.txt_mode_max_stroke = (TextView) findViewById(R.id.txt_mode_max_stroke);
        this.txt_mode_depth = (TextView) findViewById(R.id.txt_mode_depth);
        this.txt_mode_smooth = (TextView) findViewById(R.id.txt_mode_smooth);
        LanguageUtil.setText(this.mActivity, this.txt_mode_speed, "speed");
        LanguageUtil.setText(this.mActivity, this.txt_mode_max_stroke, "maxStroke");
        LanguageUtil.setText(this.mActivity, this.txt_mode_depth, "Depth");
        LanguageUtil.setText(this.mActivity, this.txt_mode_smooth, "smoothness");
        int i = this.mSMSModeType;
        LinearLayout linearLayout = this.layout_mark;
        if (i == 1) {
            linearLayout.setVisibility(8);
        } else {
            linearLayout.setVisibility(0);
        }
        this.mLineChartPlay = (LineChartPlayView3) findViewById(R.id.dialog_preview_lineChartPlay);
        this.mTxtStartTime = (TextView) findViewById(R.id.dialog_preview_txt_start_time);
        this.mTxtEndTime = (TextView) findViewById(R.id.dialog_preview_txt_end_time);
        this.mSeekBar = (SeekBar) findViewById(R.id.dialog_preview_seek_bar);
        this.mImgClose = (ImageView) findViewById(R.id.dialog_preview_img_close);
        this.timeSecond = 0;
        this.lastStart = BleManager.getInstance(this.mActivity).getmConnBleDevice().getStartPosition();
        this.lastStop = BleManager.getInstance(this.mActivity).getmConnBleDevice().getStopPosition();
        if (this.mModeData.contains(Integer.toHexString(255).toUpperCase())) {
            initModeData(this.mModeData.split("G"));
        }
        this.mLineChartPlay.setData(this.mModeData);
        if (this.mLineChartPlay.getData() != null) {
            this.isPrepared = true;
            int totalTime = ModeTimeUtil.getTotalTime(this.mModeData);
            this.mTxtEndTime.setText(ModeTimeUtil.formatTime(totalTime));
            this.mSeekBar.setMax(totalTime);
            this.mLineChartPlay.setEndTime(totalTime);
        }
        this.mLineChartPlay.setOnPlayListener(this);
        this.mImgClose.setOnClickListener(this);
        this.mSeekBar.setOnSeekBarChangeListener(this);
        setCanceledOnTouchOutside(true);
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        this.mLineChartPlay.stopSync();
        this.mLineChartPlay.setOnPlayListener(null);
        BleManager.getInstance(this.mActivity).writeShutdown();
    }

    @Override // com.hismith.hismithapp.view.LineChartPlayView3.OnPlayListener
    public void onPlaying(boolean z) {
    }

    @Override // com.hismith.hismithapp.view.LineChartPlayView3.OnPlayListener
    public void onPositionRangeChange(final int i, final int i2) {
        LogUtil.e("Ble onPositionRangeChange：" + i + " - " + i2);
        int i3 = 15;
        int i4 = 30;
        if (i >= this.lastStop) {
            i4 = 15;
            i3 = 30;
        }
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.dialog.PreviewDialog.2
            @Override // java.lang.Runnable
            public void run() {
                BleManager.getInstance(PreviewDialog.this.mActivity).writeStartPosition(i);
                PreviewDialog.this.lastStart = i;
            }
        }, i3);
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.dialog.PreviewDialog.3
            @Override // java.lang.Runnable
            public void run() {
                BleManager.getInstance(PreviewDialog.this.mActivity).writeStopPosition(i2);
                PreviewDialog.this.lastStop = i2;
            }
        }, i4);
    }

    @Override // android.widget.SeekBar.OnSeekBarChangeListener
    public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
        LogUtil.w("mStartTime progress:" + i);
    }

    @Override // com.hismith.hismithapp.view.LineChartPlayView3.OnPlayListener
    public void onSmoothChange(final int i) {
        LogUtil.e("Ble onSmoothChange：" + i);
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.dialog.PreviewDialog.4
            @Override // java.lang.Runnable
            public void run() {
                BleManager.getInstance(PreviewDialog.this.mActivity).writeSmooth(i);
            }
        }, 45L);
    }

    @Override // com.hismith.hismithapp.view.LineChartPlayView3.OnPlayListener
    public void onSpeedChange(float f) {
        LogUtil.e("Ble onSpeedChange：" + f);
        if (this.mSMSModeType == 1 && ProductUtil.isSMSType()) {
            BleManager.getInstance(this.mActivity).writeModeLine1((int) f);
        } else {
            BleManager.getInstance(this.mActivity).writeSpeed((int) f);
        }
    }

    @Override // android.widget.SeekBar.OnSeekBarChangeListener
    public void onStartTrackingTouch(SeekBar seekBar) {
        this.isDownSeek = true;
    }

    @Override // android.widget.SeekBar.OnSeekBarChangeListener
    public void onStopTrackingTouch(SeekBar seekBar) {
        this.isDownSeek = false;
        if (this.isPrepared) {
            int progress = seekBar.getProgress();
            int i = progress % 5;
            if (i != 0) {
                progress -= i;
            }
            this.mLineChartPlay.setCurrentTime(progress);
            if (this.mLineChartPlay.isRun()) {
                return;
            }
            this.mLineChartPlay.startSync();
        }
    }

    @Override // com.hismith.hismithapp.view.LineChartPlayView3.OnPlayListener
    public void onTimeChange(final int i) {
        Map<Integer, String> map = this.funDataMap;
        if (map != null && map.containsKey(Integer.valueOf(i)) && i < this.mSeekBar.getMax()) {
            LogUtil.e("onTimeChange :" + i + ", has:" + this.funDataMap.get(Integer.valueOf(i)) + ", size:" + this.funDataMap.size());
            String str = this.funDataMap.get(Integer.valueOf(i));
            if (str != null) {
                try {
                    if (str.length() == 4) {
                        int intValue = Integer.valueOf(str.substring(0, 2), 16).intValue();
                        int intValue2 = Integer.valueOf(str.substring(2, 4), 16).intValue();
                        if (intValue == 255) {
                            if (intValue2 == 0) {
                                BleManager.getInstance(this.mActivity).writeCloseReversal();
                            } else if (intValue2 == 1) {
                                BleManager.getInstance(this.mActivity).writeOpenReversal();
                            }
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        this.mActivity.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.dialog.PreviewDialog.1
            @Override // java.lang.Runnable
            public void run() {
                PreviewDialog.this.mTxtStartTime.setText(ModeTimeUtil.formatTime(i));
                if (PreviewDialog.this.isDownSeek) {
                    return;
                }
                PreviewDialog.this.mSeekBar.setProgress(i);
            }
        });
    }

    @Override // com.hismith.hismithapp.view.LineChartPlayView3.OnPlayListener
    public void onVibrateChange(float f) {
        LogUtil.e("Ble onVibrateChange：" + f);
        BleManager.getInstance(this.mActivity).writeModeLine2((int) f);
    }
}
