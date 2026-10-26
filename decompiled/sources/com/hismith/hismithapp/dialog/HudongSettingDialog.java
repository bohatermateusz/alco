package com.hismith.hismithapp.dialog;

import android.animation.ValueAnimator;
import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.LinearInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.CustomSeekbar;

/* loaded from: classes3.dex */
public class HudongSettingDialog extends Dialog implements View.OnClickListener {
    private LinearLayout dialog_confirm_layout_content;
    private float downX;
    private ImageView img_back;
    private ImageView img_mode_switch;
    private boolean isModeSwitch;
    private boolean isShowHudongView;
    private ConstraintLayout layout_mode;
    private FrameLayout layout_speed_max;
    private ConstraintLayout layout_speed_safe;
    private FrameLayout layout_vibrate_max;
    private RelativeLayout layout_vibrate_safe;
    private Activity mActivity;
    private Handler mHandler;
    private int mSafeMax;
    private int mVibrateSafeMax;
    protected OnSafeMaxListener onSafeMaxListener;
    private ProgressBar progress_speed;
    private ProgressBar progress_vibrate;
    private View.OnTouchListener safeOnTouch;
    private CustomSeekbar seek_smooth;
    private TextView txt_mode_label;
    private TextView txt_no_connected;
    private TextView txt_speed_max;
    private TextView txt_vibrate_max;
    private float vibrateDownX;
    private View.OnTouchListener vibrateSafeOnTouch;

    public interface OnSafeMaxListener {
        void onSafeMax(int i);

        void onVibrateSafeMax(int i);
    }

    public HudongSettingDialog(Activity activity, int i, int i2, OnSafeMaxListener onSafeMaxListener) {
        super(activity, R.style.MyDialog);
        this.mSafeMax = 100;
        this.isShowHudongView = true;
        this.mVibrateSafeMax = 100;
        this.isModeSwitch = true;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                ((Integer) message.obj).intValue();
                return false;
            }
        });
        this.safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    HudongSettingDialog.this.downX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - HudongSettingDialog.this.downX;
                int right = HudongSettingDialog.this.progress_speed.getRight() + (HudongSettingDialog.this.layout_speed_max.getWidth() / 2);
                LogUtil.e("getRight():" + HudongSettingDialog.this.layout_speed_max.getRight() + ", width:" + HudongSettingDialog.this.progress_speed.getRight() + ", " + HudongSettingDialog.this.layout_speed_max.getWidth());
                int right2 = (int) (right - (HudongSettingDialog.this.layout_speed_max.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > HudongSettingDialog.this.progress_speed.getWidth()) {
                    right2 = HudongSettingDialog.this.progress_speed.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) HudongSettingDialog.this.layout_speed_max.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                HudongSettingDialog.this.layout_speed_max.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / HudongSettingDialog.this.progress_speed.getWidth()));
                HudongSettingDialog.this.mSafeMax = width;
                HudongSettingDialog.this.txt_speed_max.setText(width + "");
                if (HudongSettingDialog.this.onSafeMaxListener != null) {
                    HudongSettingDialog.this.onSafeMaxListener.onSafeMax(HudongSettingDialog.this.mSafeMax);
                }
                if (width < 0 || width >= BleManager.getInstance(HudongSettingDialog.this.mActivity).getmConnBleDevice().getSpeed()) {
                    return true;
                }
                BleManager.getInstance(HudongSettingDialog.this.mActivity).writeSpeed(width);
                return true;
            }
        };
        this.vibrateSafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.5
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    HudongSettingDialog.this.vibrateDownX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - HudongSettingDialog.this.vibrateDownX;
                int right = HudongSettingDialog.this.progress_vibrate.getRight() + (HudongSettingDialog.this.layout_vibrate_max.getWidth() / 2);
                LogUtil.e("getRight():" + HudongSettingDialog.this.layout_vibrate_max.getRight() + ", width:" + HudongSettingDialog.this.progress_vibrate.getRight() + ", " + HudongSettingDialog.this.layout_vibrate_max.getWidth());
                int right2 = (int) (right - (HudongSettingDialog.this.layout_vibrate_max.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > HudongSettingDialog.this.progress_vibrate.getWidth()) {
                    right2 = HudongSettingDialog.this.progress_vibrate.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) HudongSettingDialog.this.layout_vibrate_max.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                HudongSettingDialog.this.layout_vibrate_max.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / HudongSettingDialog.this.progress_vibrate.getWidth()));
                HudongSettingDialog.this.mVibrateSafeMax = width;
                HudongSettingDialog.this.txt_vibrate_max.setText(width + "");
                if (HudongSettingDialog.this.onSafeMaxListener != null) {
                    HudongSettingDialog.this.onSafeMaxListener.onVibrateSafeMax(HudongSettingDialog.this.mVibrateSafeMax);
                }
                if (width < 0 || width >= BleManager.getInstance(HudongSettingDialog.this.mActivity).getmConnBleDevice().getVibration()) {
                    return true;
                }
                BleManager.getInstance(HudongSettingDialog.this.mActivity).writeVibrate(width);
                return true;
            }
        };
        this.mActivity = activity;
        this.mSafeMax = i;
        this.mVibrateSafeMax = i2;
        this.onSafeMaxListener = onSafeMaxListener;
    }

    public HudongSettingDialog(Activity activity, int i, boolean z, int i2, OnSafeMaxListener onSafeMaxListener) {
        super(activity, R.style.MyDialog);
        this.mSafeMax = 100;
        this.isShowHudongView = true;
        this.mVibrateSafeMax = 100;
        this.isModeSwitch = true;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                ((Integer) message.obj).intValue();
                return false;
            }
        });
        this.safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.4
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    HudongSettingDialog.this.downX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - HudongSettingDialog.this.downX;
                int right = HudongSettingDialog.this.progress_speed.getRight() + (HudongSettingDialog.this.layout_speed_max.getWidth() / 2);
                LogUtil.e("getRight():" + HudongSettingDialog.this.layout_speed_max.getRight() + ", width:" + HudongSettingDialog.this.progress_speed.getRight() + ", " + HudongSettingDialog.this.layout_speed_max.getWidth());
                int right2 = (int) (right - (HudongSettingDialog.this.layout_speed_max.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > HudongSettingDialog.this.progress_speed.getWidth()) {
                    right2 = HudongSettingDialog.this.progress_speed.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) HudongSettingDialog.this.layout_speed_max.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                HudongSettingDialog.this.layout_speed_max.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / HudongSettingDialog.this.progress_speed.getWidth()));
                HudongSettingDialog.this.mSafeMax = width;
                HudongSettingDialog.this.txt_speed_max.setText(width + "");
                if (HudongSettingDialog.this.onSafeMaxListener != null) {
                    HudongSettingDialog.this.onSafeMaxListener.onSafeMax(HudongSettingDialog.this.mSafeMax);
                }
                if (width < 0 || width >= BleManager.getInstance(HudongSettingDialog.this.mActivity).getmConnBleDevice().getSpeed()) {
                    return true;
                }
                BleManager.getInstance(HudongSettingDialog.this.mActivity).writeSpeed(width);
                return true;
            }
        };
        this.vibrateSafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.5
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    HudongSettingDialog.this.vibrateDownX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - HudongSettingDialog.this.vibrateDownX;
                int right = HudongSettingDialog.this.progress_vibrate.getRight() + (HudongSettingDialog.this.layout_vibrate_max.getWidth() / 2);
                LogUtil.e("getRight():" + HudongSettingDialog.this.layout_vibrate_max.getRight() + ", width:" + HudongSettingDialog.this.progress_vibrate.getRight() + ", " + HudongSettingDialog.this.layout_vibrate_max.getWidth());
                int right2 = (int) (right - (HudongSettingDialog.this.layout_vibrate_max.getRight() + x));
                if (right2 < 0) {
                    right2 = 0;
                } else if (right2 > HudongSettingDialog.this.progress_vibrate.getWidth()) {
                    right2 = HudongSettingDialog.this.progress_vibrate.getWidth();
                }
                LogUtil.e("safe right:" + right2);
                RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) HudongSettingDialog.this.layout_vibrate_max.getLayoutParams();
                layoutParams.setMargins(0, 0, right2, 0);
                HudongSettingDialog.this.layout_vibrate_max.setLayoutParams(layoutParams);
                int width = (int) (100.0d - ((right2 * 100.0d) / HudongSettingDialog.this.progress_vibrate.getWidth()));
                HudongSettingDialog.this.mVibrateSafeMax = width;
                HudongSettingDialog.this.txt_vibrate_max.setText(width + "");
                if (HudongSettingDialog.this.onSafeMaxListener != null) {
                    HudongSettingDialog.this.onSafeMaxListener.onVibrateSafeMax(HudongSettingDialog.this.mVibrateSafeMax);
                }
                if (width < 0 || width >= BleManager.getInstance(HudongSettingDialog.this.mActivity).getmConnBleDevice().getVibration()) {
                    return true;
                }
                BleManager.getInstance(HudongSettingDialog.this.mActivity).writeVibrate(width);
                return true;
            }
        };
        this.mActivity = activity;
        this.mSafeMax = i;
        this.mVibrateSafeMax = i2;
        this.onSafeMaxListener = onSafeMaxListener;
        this.isShowHudongView = z;
    }

    private void initBleData() {
        try {
            Object obj = SharedPreferencesUtil.get(this.mActivity, Constans.RemoteShareMode, true);
            if (obj != null) {
                this.isModeSwitch = ((Boolean) obj).booleanValue();
            }
            boolean z = this.isModeSwitch;
            ImageView imageView = this.img_mode_switch;
            if (z) {
                imageView.setImageResource(R.mipmap.ic_setting_switch_on);
            } else {
                imageView.setImageResource(R.mipmap.ic_setting_switch_off);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        refreshSafeUI(this.mSafeMax);
        refreshVibrateSafeUI(this.mVibrateSafeMax);
        if (BleManager.getInstance(this.mActivity).getmConnBleDevice() != null) {
            refreshSmoothUI(BleManager.getInstance(this.mActivity).getmConnBleDevice().getSmoothness());
        }
    }

    private void initProduct() {
        ProductModel productModel = MyApplication.getApplication().getmProductModel();
        LinearLayout linearLayout = this.dialog_confirm_layout_content;
        if (productModel == null) {
            linearLayout.setVisibility(8);
            this.txt_no_connected.setVisibility(0);
            return;
        }
        linearLayout.setVisibility(0);
        this.txt_no_connected.setVisibility(8);
        if (!this.isShowHudongView) {
            this.layout_speed_safe.setVisibility(8);
            this.layout_vibrate_safe.setVisibility(8);
            this.seek_smooth.setVisibility(8);
            return;
        }
        if (Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
            this.layout_speed_safe.setVisibility(0);
            this.layout_vibrate_safe.setVisibility(8);
            this.seek_smooth.setVisibility(0);
            return;
        }
        boolean equals = Constans.DEVICE_TYPE_SMV.equals(MyApplication.getApplication().getmProductModel().getDeviceType());
        ConstraintLayout constraintLayout = this.layout_speed_safe;
        if (equals) {
            constraintLayout.setVisibility(0);
            this.layout_vibrate_safe.setVisibility(0);
            this.seek_smooth.setVisibility(8);
        } else {
            constraintLayout.setVisibility(0);
            this.layout_vibrate_safe.setVisibility(8);
            this.seek_smooth.setVisibility(8);
        }
    }

    private void refreshSafeUI(int i) {
        this.txt_speed_max.setText(i + "");
        double width = this.progress_speed.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_speed_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_speed_max.setLayoutParams(layoutParams);
    }

    private void refreshSmoothUI(int i) {
        ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.3
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                HudongSettingDialog.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
            }
        });
        ofInt.start();
    }

    private void refreshVibrateSafeUI(int i) {
        this.txt_vibrate_max.setText(i + "");
        double width = this.progress_vibrate.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.layout_vibrate_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_vibrate_max.setLayoutParams(layoutParams);
    }

    private void setListener() {
        this.img_back.setOnClickListener(this);
        this.img_mode_switch.setOnClickListener(this);
        this.layout_speed_max.setOnTouchListener(this.safeOnTouch);
        this.layout_vibrate_max.setOnTouchListener(this.vibrateSafeOnTouch);
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.2
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + HudongSettingDialog.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + HudongSettingDialog.this.seek_smooth.getProgress());
                int progress = HudongSettingDialog.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                BleManager.getInstance(HudongSettingDialog.this.mActivity).writeSmooth(i / 100);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.2.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        HudongSettingDialog.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
    }

    @Override // android.app.Dialog, android.content.DialogInterface
    public void dismiss() {
        super.dismiss();
        LogUtil.e("消失");
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.img_back) {
            dismiss();
            return;
        }
        if (id == R.id.img_mode_switch) {
            boolean z = this.isModeSwitch;
            this.isModeSwitch = !z;
            ImageView imageView = this.img_mode_switch;
            if (z) {
                imageView.setImageResource(R.mipmap.ic_setting_switch_off);
                BleManager.getInstance(this.mActivity).writeSpeed(BleManager.getInstance(this.mActivity).getmConnBleDevice().getSpeed());
                if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getFuns() != null && MyApplication.getApplication().getmProductModel().getFuns().length > 1) {
                    new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.dialog.HudongSettingDialog.6
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(HudongSettingDialog.this.mActivity).writeVibrate(BleManager.getInstance(HudongSettingDialog.this.mActivity).getmConnBleDevice().getVibration());
                        }
                    }, 100L);
                }
            } else {
                imageView.setImageResource(R.mipmap.ic_setting_switch_on);
            }
            SocketManager.getInstance().switchMode(this.isModeSwitch ? 1 : 0);
            SharedPreferencesUtil.put(this.mActivity, Constans.RemoteShareMode, Boolean.valueOf(this.isModeSwitch));
        }
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_hu_dong_setting);
        Window window = getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (this.mActivity.getWindowManager().getDefaultDisplay().getWidth() * 0.9d);
            window.setAttributes(attributes);
        }
        this.img_back = (ImageView) findViewById(R.id.img_back);
        this.dialog_confirm_layout_content = (LinearLayout) findViewById(R.id.dialog_confirm_layout_content);
        TextView textView = (TextView) findViewById(R.id.txt_no_connected);
        this.txt_no_connected = textView;
        LanguageUtil.setText(this.mActivity, textView, "noConnectedDevice");
        this.layout_mode = (ConstraintLayout) findViewById(R.id.layout_mode);
        TextView textView2 = (TextView) findViewById(R.id.txt_mode_label);
        this.txt_mode_label = textView2;
        LanguageUtil.setText(this.mActivity, textView2, "modeControl");
        this.img_mode_switch = (ImageView) findViewById(R.id.img_mode_switch);
        this.layout_speed_safe = (ConstraintLayout) findViewById(R.id.layout_speed_safe);
        this.progress_speed = (ProgressBar) findViewById(R.id.progress_speed);
        this.layout_speed_max = (FrameLayout) findViewById(R.id.layout_speed_max);
        this.txt_speed_max = (TextView) findViewById(R.id.txt_speed_max);
        this.layout_vibrate_safe = (RelativeLayout) findViewById(R.id.layout_vibrate_safe);
        this.progress_vibrate = (ProgressBar) findViewById(R.id.progress_vibrate);
        this.layout_vibrate_max = (FrameLayout) findViewById(R.id.layout_vibrate_max);
        this.txt_vibrate_max = (TextView) findViewById(R.id.txt_vibrate_max);
        this.seek_smooth = (CustomSeekbar) findViewById(R.id.seek_smooth);
        initProduct();
        setCanceledOnTouchOutside(true);
        setListener();
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        LogUtil.e("消失onDetachedFromWindow");
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        if (z) {
            initBleData();
        }
    }

    public void setOnSafeMaxListener(OnSafeMaxListener onSafeMaxListener) {
        this.onSafeMaxListener = onSafeMaxListener;
    }

    @Override // android.app.Dialog
    public void show() {
        super.show();
        LogUtil.e("显示");
    }
}
