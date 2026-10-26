package com.hismith.hismithapp.view.plug;

import android.animation.ValueAnimator;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.LinearInterpolator;
import android.widget.SeekBar;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.view.CustomSeekbar;

/* loaded from: classes3.dex */
public class PlugSMSSmooth {
    private View baseView;
    private Context mContext;
    private ViewGroup rootView;
    private CustomSeekbar seek_smooth;

    public PlugSMSSmooth(Context context, ViewGroup viewGroup) {
        this.mContext = context;
        this.rootView = viewGroup;
        View inflate = LayoutInflater.from(context).inflate(R.layout.plug_sms_smooth, viewGroup, false);
        this.baseView = inflate;
        ViewGroup.MarginLayoutParams marginLayoutParams = (ViewGroup.MarginLayoutParams) inflate.getLayoutParams();
        marginLayoutParams.setMargins(0, MeasureUtil.dip2px(this.mContext, 10.0f), 0, 0);
        this.baseView.setLayoutParams(marginLayoutParams);
        initView();
        init();
        setListener();
    }

    private void init() {
    }

    private void initView() {
        this.seek_smooth = (CustomSeekbar) this.baseView.findViewById(R.id.seek_smooth);
    }

    private void refreshSmoothUI(int i) {
        ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.plug.PlugSMSSmooth.2
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                PlugSMSSmooth.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
            }
        });
        ofInt.start();
    }

    private void setListener() {
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.view.plug.PlugSMSSmooth.1
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + PlugSMSSmooth.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + PlugSMSSmooth.this.seek_smooth.getProgress());
                int progress = PlugSMSSmooth.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                BleManager.getInstance(PlugSMSSmooth.this.mContext).writeSmooth(i / 100);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.view.plug.PlugSMSSmooth.1.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        PlugSMSSmooth.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
    }

    public View getBaseView() {
        return this.baseView;
    }

    public int getSmooth() {
        return this.seek_smooth.getProgress() / 100;
    }

    public void setSmooth(int i) {
        refreshSmoothUI(i);
    }
}
