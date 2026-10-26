package com.hismith.hismithapp.view;

import android.app.Activity;
import android.view.View;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.lxj.xpopup.core.CenterPopupView;

/* loaded from: classes3.dex */
public class SpeedAttachPopup extends CenterPopupView {
    private Activity mActivity;
    private ImageView mImgConfirm;
    private OnSubmitListener mOnSubmitListener;
    private SeekBar mSeekSpeed;
    private TextView mTxtMax;
    private TextView mTxtMin;
    private TextView mTxtSpeed;
    private TextView mTxtSpeedTitle;
    private int mValue;

    public interface OnSubmitListener {
        void onSubmit(int i);
    }

    public SpeedAttachPopup(Activity activity, int i, OnSubmitListener onSubmitListener) {
        super(activity);
        this.mActivity = activity;
        this.mValue = i;
        this.mOnSubmitListener = onSubmitListener;
    }

    @Override // com.lxj.xpopup.core.CenterPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.popup_speed;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        this.mTxtSpeedTitle = (TextView) findViewById(R.id.popup_speed_txt_speedTitle);
        this.mTxtSpeed = (TextView) findViewById(R.id.popup_speed_txt_speed);
        this.mTxtMin = (TextView) findViewById(R.id.popup_speed_txt_min);
        this.mTxtMax = (TextView) findViewById(R.id.popup_speed_txt_max);
        this.mImgConfirm = (ImageView) findViewById(R.id.popup_speed_img_confirm);
        this.mSeekSpeed = (SeekBar) findViewById(R.id.popup_speed_seek_speed);
        LanguageUtil.setText(this.mActivity, this.mTxtSpeedTitle, "speedTitle");
        this.mTxtMin.setText("0");
        this.mTxtMax.setText(this.mSeekSpeed.getMax() + "");
        this.mTxtSpeed.setText(this.mValue + "");
        this.mSeekSpeed.setProgress(this.mValue);
        this.mImgConfirm.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.view.SpeedAttachPopup.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (SpeedAttachPopup.this.mOnSubmitListener != null) {
                    SpeedAttachPopup.this.mOnSubmitListener.onSubmit(SpeedAttachPopup.this.mSeekSpeed.getProgress());
                }
                SpeedAttachPopup.this.dismiss();
            }
        });
        this.mSeekSpeed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.view.SpeedAttachPopup.2
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                SpeedAttachPopup.this.mTxtSpeed.setText(i + "");
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });
    }

    @Override // com.lxj.xpopup.core.BasePopupView, android.view.ViewGroup, android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        OnSubmitListener onSubmitListener = this.mOnSubmitListener;
        if (onSubmitListener != null) {
            onSubmitListener.onSubmit(this.mSeekSpeed.getProgress());
        }
    }
}
