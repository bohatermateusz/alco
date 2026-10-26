package com.hismith.hismithapp.fragment;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.fragment.app.Fragment;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.surface_view.LineDataUtil;
import com.hismith.hismithapp.surface_view.LineDrawView6;
import com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ModeTimeUtil;

/* loaded from: classes3.dex */
public class PreviewFragment2 extends Fragment implements View.OnClickListener, OnPlayTimeListener, SeekBar.OnSeekBarChangeListener {
    private boolean isPlay;
    private LinearLayout layout_mark;
    private Activity mActivity;
    private ImageView mImgClose;
    private ImageView mImgPause;
    private LineDrawView6 mLineDrawView;
    private int mModeId;
    private String mModePoints;
    private int mPlayTime;
    private int mSafe;
    private SeekBar mSeekBar;
    private TextView mTxtEndTime;
    private TextView mTxtStartTime;
    private int mVibrateSafe;
    private OnCloseClickListener onCloseClickListener;
    private OnSpeedChangeListener onSpeedChangeListener;
    private TextView txt_mode_depth;
    private TextView txt_mode_max_stroke;
    private TextView txt_mode_smooth;
    private TextView txt_mode_speed;
    private int lastSpeed = 0;
    private int lastVibrate = 0;
    private int lastStartPosition = 0;
    private int lastStopPosition = 0;
    private int lastSmooth = 0;
    private int mSMSModeType = 1;
    private boolean isPrepared = false;
    private boolean isDownSeek = false;
    private boolean isAllowClose = false;
    private boolean isAllowTouchSeekBar = false;
    private Handler mHandler = new Handler(Looper.getMainLooper());
    private final int delay = 15;

    public interface OnCloseClickListener {
        void onCloseClick(int i);
    }

    public interface OnSpeedChangeListener {
        void onSpeedChange(int i);

        void onVibrateChange(int i);
    }

    public PreviewFragment2(int i, String str, int i2, int i3, int i4, boolean z) {
        this.mSafe = 100;
        this.mVibrateSafe = 100;
        this.isPlay = true;
        this.mModeId = i;
        this.mModePoints = str;
        this.mPlayTime = i2;
        this.mSafe = i3;
        this.mVibrateSafe = i4;
        this.isPlay = z;
    }

    public void attachedToStartRun() {
        LineDrawView6 lineDrawView6 = this.mLineDrawView;
        if (lineDrawView6 != null) {
            lineDrawView6.attachedToStartRun();
        }
    }

    public void detachedToStopRun() {
        LineDrawView6 lineDrawView6 = this.mLineDrawView;
        if (lineDrawView6 != null) {
            lineDrawView6.detachedToStopRun();
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onActivityCreated(Bundle bundle) {
        LineDrawView6 lineDrawView6;
        super.onActivityCreated(bundle);
        if (this.isPrepared && this.isPlay && (lineDrawView6 = this.mLineDrawView) != null) {
            lineDrawView6.play();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.dialog_preview_img_close) {
            OnCloseClickListener onCloseClickListener = this.onCloseClickListener;
            if (onCloseClickListener != null) {
                onCloseClickListener.onCloseClick(this.mSMSModeType);
                return;
            }
            return;
        }
        if (id == R.id.dialog_preview_img_pause) {
            if (this.isPlay) {
                setPlay(false);
                SocketManager.getInstance().roomStopMode();
            } else {
                setPlay(true);
                SocketManager.getInstance().roomSetMode(this.mModeId, this.mLineDrawView.getmPlayTime() / 1000);
            }
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onComplete() {
        OnCloseClickListener onCloseClickListener = this.onCloseClickListener;
        if (onCloseClickListener != null) {
            onCloseClickListener.onCloseClick(this.mSMSModeType);
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mActivity = getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        String buildToUIData;
        String buildToUIData2;
        String str;
        String str2;
        View inflate = layoutInflater.inflate(R.layout.dialog_preview2, (ViewGroup) null);
        this.layout_mark = (LinearLayout) inflate.findViewById(R.id.layout_mark);
        this.txt_mode_speed = (TextView) inflate.findViewById(R.id.txt_mode_speed);
        this.txt_mode_max_stroke = (TextView) inflate.findViewById(R.id.txt_mode_max_stroke);
        this.txt_mode_depth = (TextView) inflate.findViewById(R.id.txt_mode_depth);
        this.txt_mode_smooth = (TextView) inflate.findViewById(R.id.txt_mode_smooth);
        LanguageUtil.setText(this.mActivity, this.txt_mode_speed, "speed");
        LanguageUtil.setText(this.mActivity, this.txt_mode_max_stroke, "maxStroke");
        LanguageUtil.setText(this.mActivity, this.txt_mode_depth, "Depth");
        LanguageUtil.setText(this.mActivity, this.txt_mode_smooth, "smoothness");
        this.mLineDrawView = (LineDrawView6) inflate.findViewById(R.id.dialog_preview_lineDrawView);
        this.mTxtStartTime = (TextView) inflate.findViewById(R.id.dialog_preview_txt_start_time);
        this.mTxtEndTime = (TextView) inflate.findViewById(R.id.dialog_preview_txt_end_time);
        this.mSeekBar = (SeekBar) inflate.findViewById(R.id.dialog_preview_seek_bar);
        this.mImgClose = (ImageView) inflate.findViewById(R.id.dialog_preview_img_close);
        this.mImgPause = (ImageView) inflate.findViewById(R.id.dialog_preview_img_pause);
        this.mLineDrawView.setPlayCircle(true);
        this.mLineDrawView.setDrawYScale(false);
        boolean z = this.isAllowClose;
        ImageView imageView = this.mImgClose;
        if (z) {
            imageView.setVisibility(0);
        } else {
            imageView.setVisibility(8);
        }
        boolean z2 = this.isPlay;
        ImageView imageView2 = this.mImgPause;
        if (z2) {
            if (imageView2 != null) {
                imageView2.setImageResource(R.mipmap.ic_mode_pause);
            }
        } else if (imageView2 != null) {
            imageView2.setImageResource(R.mipmap.ic_mode_play);
        }
        boolean z3 = this.isAllowTouchSeekBar;
        SeekBar seekBar = this.mSeekBar;
        if (z3) {
            seekBar.setEnabled(true);
            this.mImgPause.setOnClickListener(this);
        } else {
            seekBar.setEnabled(false);
            this.mImgPause.setOnClickListener(null);
        }
        String[] buildServerToClientData = LineDataUtil.buildServerToClientData(this.mModePoints);
        if (!TextUtils.isEmpty(buildServerToClientData[0])) {
            int parseInt = Integer.parseInt(buildServerToClientData[0]);
            if (buildServerToClientData.length == 5) {
                this.mSMSModeType = 2;
                this.layout_mark.setVisibility(0);
                buildToUIData = LineDataUtil.buildToUIData(buildServerToClientData[4], this.mLineDrawView.getIntervalTime(), parseInt);
                str = LineDataUtil.buildToUIData(buildServerToClientData[1], this.mLineDrawView.getIntervalTime(), parseInt);
                buildToUIData2 = LineDataUtil.buildToUIData(buildServerToClientData[2], this.mLineDrawView.getIntervalTime(), parseInt);
                str2 = LineDataUtil.buildToUIData(buildServerToClientData[3], this.mLineDrawView.getIntervalTime(), parseInt);
            } else {
                this.mSMSModeType = 1;
                this.layout_mark.setVisibility(8);
                buildToUIData = LineDataUtil.buildToUIData(buildServerToClientData[1], this.mLineDrawView.getIntervalTime(), parseInt);
                buildToUIData2 = LineDataUtil.buildToUIData(buildServerToClientData[2], this.mLineDrawView.getIntervalTime(), parseInt);
                str = "";
                str2 = "";
            }
            boolean isEmpty = TextUtils.isEmpty(buildToUIData2);
            LineDrawView6 lineDrawView6 = this.mLineDrawView;
            if (isEmpty) {
                lineDrawView6.setVibrateLine(false);
            } else {
                lineDrawView6.setVibrateLine(true);
            }
            boolean isEmpty2 = TextUtils.isEmpty(str);
            LineDrawView6 lineDrawView62 = this.mLineDrawView;
            if (isEmpty2) {
                lineDrawView62.setStartPositionLine(false);
            } else {
                lineDrawView62.setStartPositionLine(true);
            }
            boolean isEmpty3 = TextUtils.isEmpty(str2);
            LineDrawView6 lineDrawView63 = this.mLineDrawView;
            if (isEmpty3) {
                lineDrawView63.setSmoothLine(false);
            } else {
                lineDrawView63.setSmoothLine(true);
            }
            this.mLineDrawView.setData(buildToUIData, buildToUIData2, str, str2);
        }
        if (ProductUtil.isSMSType() && this.mSMSModeType == 1) {
            BleManager.getInstance(this.mActivity).writeRunManual();
        }
        if (!this.mLineDrawView.isDataEmpty()) {
            this.isPrepared = true;
            this.mTxtEndTime.setText(ModeTimeUtil.formatMilliTime(this.mLineDrawView.getmTime()));
            this.mSeekBar.setMax(this.mLineDrawView.getmTime());
            this.mLineDrawView.setPlayTime(this.mPlayTime);
        }
        this.mLineDrawView.setOnPlayTimeListener(this);
        this.mImgClose.setOnClickListener(this);
        this.mSeekBar.setOnSeekBarChangeListener(this);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroyView() {
        super.onDestroyView();
        LineDrawView6 lineDrawView6 = this.mLineDrawView;
        if (lineDrawView6 != null) {
            lineDrawView6.pause();
            this.mLineDrawView.setOnPlayTimeListener(null);
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onFunData(String str, String str2) {
        int intValue = Integer.valueOf(str, 16).intValue();
        int intValue2 = Integer.valueOf(str2, 16).intValue();
        if (intValue == 254) {
            if (intValue2 == 1) {
                BleManager.getInstance(this.mActivity).writeOpenPiu();
            }
        } else {
            if (intValue != 255) {
                return;
            }
            if (intValue2 == 0) {
                BleManager.getInstance(this.mActivity).writeCloseReversal();
            } else if (intValue2 == 1) {
                BleManager.getInstance(this.mActivity).writeOpenReversal();
            }
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onLinePause() {
        if (ProductUtil.isSMSType() && this.mSMSModeType == 1) {
            return;
        }
        BleManager.getInstance(this.mActivity).writeSpeed(0);
        this.lastSpeed = 0;
        new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.PreviewFragment2.4
            @Override // java.lang.Runnable
            public void run() {
                BleManager.getInstance(PreviewFragment2.this.mActivity).writeModeLine2(0);
                PreviewFragment2.this.lastVibrate = 0;
            }
        }, 100L);
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onPlayTime(int i) {
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
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.PreviewFragment2.1
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(PreviewFragment2.this.mActivity).writeData(BleManager.getInstance(PreviewFragment2.this.mActivity).getBleCCData(7, i));
                    PreviewFragment2.this.lastStartPosition = i;
                }
            }, i3);
        }
        if (i2 != this.lastStopPosition) {
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.PreviewFragment2.2
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(PreviewFragment2.this.mActivity).writeData(BleManager.getInstance(PreviewFragment2.this.mActivity).getBleCCData(8, i2));
                    PreviewFragment2.this.lastStopPosition = i2;
                }
            }, i4);
        }
    }

    @Override // android.widget.SeekBar.OnSeekBarChangeListener
    public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onSmoothChange(final int i) {
        if (i != this.lastSmooth) {
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.PreviewFragment2.3
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(PreviewFragment2.this.mActivity).writeData(BleManager.getInstance(PreviewFragment2.this.mActivity).getBleCCData(9, i));
                    PreviewFragment2.this.lastSmooth = i;
                }
            }, 45L);
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onSpeedChange(int i) {
        int i2;
        if ((!ProductUtil.isSMSType() || this.mSMSModeType != 1) && i > (i2 = this.mSafe)) {
            i = i2;
        }
        LogUtil.e("mSafe:" + this.mSafe + ", speed:" + i);
        OnSpeedChangeListener onSpeedChangeListener = this.onSpeedChangeListener;
        if (onSpeedChangeListener != null) {
            onSpeedChangeListener.onSpeedChange(i);
        }
        if (i != this.lastSpeed) {
            this.lastSpeed = i;
            if (this.mSMSModeType == 1 && ProductUtil.isSMSType()) {
                BleManager.getInstance(this.mActivity).writePosition(i);
            } else {
                BleManager.getInstance(this.mActivity).writeSpeed(i);
            }
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
            setPlay(true);
            int progress = seekBar.getProgress();
            int i = progress % 5;
            if (i != 0) {
                progress -= i;
            }
            this.mLineDrawView.setPlayTime(progress);
            int i2 = progress / 1000;
            if (this.isAllowTouchSeekBar) {
                SocketManager.getInstance().roomSetMode(this.mModeId, i2);
            }
            if (this.mLineDrawView.isPlay()) {
                return;
            }
            this.mLineDrawView.play();
        }
    }

    @Override // com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener
    public void onVibrateChange(int i) {
        int i2 = this.mVibrateSafe;
        if (i > i2) {
            i = i2;
        }
        LogUtil.e("mVibrateSafe:" + this.mVibrateSafe + ", vibrate:" + i);
        OnSpeedChangeListener onSpeedChangeListener = this.onSpeedChangeListener;
        if (onSpeedChangeListener != null) {
            onSpeedChangeListener.onVibrateChange(i);
        }
        if (i != this.lastVibrate) {
            BleManager.getInstance(this.mActivity).writeModeLine2(i);
            this.lastVibrate = i;
        }
    }

    public void setAllowClose(Boolean bool) {
        boolean booleanValue = bool.booleanValue();
        this.isAllowClose = booleanValue;
        ImageView imageView = this.mImgClose;
        if (booleanValue) {
            if (imageView != null) {
                imageView.setVisibility(0);
            }
        } else if (imageView != null) {
            imageView.setVisibility(8);
        }
    }

    public void setOnCloseClickListener(OnCloseClickListener onCloseClickListener) {
        this.onCloseClickListener = onCloseClickListener;
    }

    public void setOnSpeedChangeListener(OnSpeedChangeListener onSpeedChangeListener) {
        this.onSpeedChangeListener = onSpeedChangeListener;
    }

    public void setPlay(boolean z) {
        this.isPlay = z;
        LineDrawView6 lineDrawView6 = this.mLineDrawView;
        if (!z) {
            if (lineDrawView6 != null && lineDrawView6.isPlay()) {
                this.mLineDrawView.pause();
            }
            ImageView imageView = this.mImgPause;
            if (imageView != null) {
                imageView.setImageResource(R.mipmap.ic_mode_play);
                return;
            }
            return;
        }
        if (lineDrawView6 != null && !lineDrawView6.isPlay() && this.isPrepared) {
            this.mLineDrawView.play();
        }
        ImageView imageView2 = this.mImgPause;
        if (imageView2 != null) {
            imageView2.setImageResource(R.mipmap.ic_mode_pause);
        }
    }

    public void setSeekBarClickable(boolean z) {
        this.isAllowTouchSeekBar = z;
        SeekBar seekBar = this.mSeekBar;
        if (z) {
            if (seekBar != null) {
                seekBar.setEnabled(true);
            }
            ImageView imageView = this.mImgPause;
            if (imageView != null) {
                imageView.setOnClickListener(this);
                return;
            }
            return;
        }
        if (seekBar != null) {
            seekBar.setEnabled(false);
        }
        ImageView imageView2 = this.mImgPause;
        if (imageView2 != null) {
            imageView2.setOnClickListener(null);
        }
    }

    public void setmPlayTime(int i) {
        this.mPlayTime = i;
        if (!this.isPlay) {
            setPlay(true);
        }
        if (this.isPrepared) {
            this.mLineDrawView.setPlayTime(i);
        }
    }

    public void setmSafe(int i) {
        this.mSafe = i;
    }

    public void setmVibrateSafe(int i) {
        this.mVibrateSafe = i;
    }
}
