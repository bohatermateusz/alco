package com.hismith.hismithapp.dialog;

import android.animation.ValueAnimator;
import android.app.Activity;
import android.app.Dialog;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.LinearInterpolator;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.SettingProgressView;

/* loaded from: classes3.dex */
public class PlayModeSettingDialog extends Dialog implements View.OnClickListener {
    private LinearLayout dialog_confirm_layout_content;
    private float endDownX;
    private View.OnTouchListener endOnTouch;
    private ImageView img_back;
    private ConstraintLayout layout_position;
    private ConstraintLayout layout_position_max;
    private ConstraintLayout layout_position_min;
    private LinearLayout layout_slide_speed;
    private ConstraintLayout layout_smooth;
    private Activity mActivity;
    private Handler mHandler;
    private onStatusListener mOnStatusListener;
    private int mPositionEnd;
    private int mPositionStart;
    private boolean mSMSUIShow;
    private boolean mSlideShow;
    private ProgressBar progress_position;
    private SettingProgressView progress_slide_speed;
    private CustomSeekbar seek_smooth;
    private float startDownX;
    private View.OnTouchListener startOnTouch;
    private int start_stop_position_diff;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private TextView txt_slide_speed;

    public interface onStatusListener {
        void onDismiss();

        void onShow();
    }

    public PlayModeSettingDialog(Activity activity, onStatusListener onstatuslistener, boolean z, boolean z2) {
        super(activity, R.style.MyDialog);
        this.mSlideShow = false;
        this.mSMSUIShow = false;
        this.start_stop_position_diff = Constans.start_stop_position_diff;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.PlayModeSettingDialog.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int intValue = ((Integer) message.obj).intValue();
                int i = message.what;
                if (i == 7) {
                    PlayModeSettingDialog.this.refreshStartBitUI(intValue);
                    return false;
                }
                if (i != 8) {
                    return false;
                }
                PlayModeSettingDialog.this.refreshEndBitUI(intValue);
                return false;
            }
        });
        this.mPositionEnd = 100;
        this.endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.dialog.PlayModeSettingDialog.5
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    view.bringToFront();
                    PlayModeSettingDialog.this.endDownX = motionEvent.getX();
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayModeSettingDialog.this.endDownX;
                int width = PlayModeSettingDialog.this.progress_position.getWidth();
                int right = (int) ((PlayModeSettingDialog.this.progress_position.getRight() + (PlayModeSettingDialog.this.layout_position_max.getWidth() / 2)) - (PlayModeSettingDialog.this.layout_position_max.getRight() + x));
                LogUtil.e("onTouch moveX:" + x + ", right():" + right);
                if (right < 0) {
                    right = 0;
                } else if (right > width) {
                    right = width;
                }
                double d = width;
                int i = (int) (100.0d - ((right * 100.0d) / d));
                if (i <= PlayModeSettingDialog.this.mPositionStart + PlayModeSettingDialog.this.start_stop_position_diff) {
                    int i2 = PlayModeSettingDialog.this.mPositionStart + PlayModeSettingDialog.this.start_stop_position_diff;
                    i = i2 > 100 ? 100 : i2;
                    right = (int) (d - ((d / 100.0d) * i));
                }
                int i3 = i <= 100 ? i < 0 ? 0 : i : 100;
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayModeSettingDialog.this.layout_position_max.getLayoutParams();
                layoutParams.setMargins(0, 0, right, 0);
                PlayModeSettingDialog.this.layout_position_max.setLayoutParams(layoutParams);
                if (PlayModeSettingDialog.this.mPositionEnd == i3) {
                    return true;
                }
                PlayModeSettingDialog.this.mPositionEnd = i3;
                LogUtil.e("结束位置：" + PlayModeSettingDialog.this.mPositionEnd + ", right:" + right);
                PlayModeSettingDialog.this.txt_position_max.setText(i3 + "");
                BleManager.getInstance(PlayModeSettingDialog.this.mActivity).writeStopPosition(PlayModeSettingDialog.this.mPositionEnd);
                return true;
            }
        };
        this.mPositionStart = 0;
        this.startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.dialog.PlayModeSettingDialog.6
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int action = motionEvent.getAction();
                if (action == 0) {
                    view.bringToFront();
                    PlayModeSettingDialog.this.startDownX = motionEvent.getX();
                    LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                    LogUtil.e("onTouch left:" + view.getLeft());
                    return true;
                }
                if (action != 2) {
                    return true;
                }
                float x = motionEvent.getX() - PlayModeSettingDialog.this.startDownX;
                LogUtil.e("抖动：" + motionEvent.getX() + ", " + PlayModeSettingDialog.this.startDownX + " = " + x);
                int width = PlayModeSettingDialog.this.progress_position.getWidth();
                int left = (int) ((PlayModeSettingDialog.this.layout_position_min.getLeft() + x) - (PlayModeSettingDialog.this.progress_position.getLeft() - (PlayModeSettingDialog.this.layout_position_min.getWidth() / 2)));
                LogUtil.e("onTouch moveX:" + x + ", left():" + left);
                if (left < 0) {
                    left = 0;
                } else if (left > width) {
                    left = width;
                }
                double d = width;
                int i = (int) ((left * 100.0d) / d);
                if (i >= PlayModeSettingDialog.this.mPositionEnd - PlayModeSettingDialog.this.start_stop_position_diff) {
                    int i2 = PlayModeSettingDialog.this.mPositionEnd - PlayModeSettingDialog.this.start_stop_position_diff;
                    i = i2 < 0 ? 0 : i2;
                    left = (int) ((d / 100.0d) * i);
                }
                if (i > 100) {
                    i = 100;
                } else if (i < 0) {
                    i = 0;
                }
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlayModeSettingDialog.this.layout_position_min.getLayoutParams();
                layoutParams.setMargins(left, 0, 0, 0);
                PlayModeSettingDialog.this.layout_position_min.setLayoutParams(layoutParams);
                if (PlayModeSettingDialog.this.mPositionStart == i) {
                    return true;
                }
                PlayModeSettingDialog.this.mPositionStart = i;
                LogUtil.e("开始位置：" + PlayModeSettingDialog.this.mPositionStart + ", left:" + left);
                PlayModeSettingDialog.this.txt_position_min.setText(i + "");
                BleManager.getInstance(PlayModeSettingDialog.this.mActivity).writeStartPosition(PlayModeSettingDialog.this.mPositionStart);
                return true;
            }
        };
        this.mActivity = activity;
        this.mOnStatusListener = onstatuslistener;
        this.mSlideShow = z;
        this.mSMSUIShow = z2;
    }

    private void initBleData() {
        if (BleManager.getInstance(this.mActivity).getmConnBleDevice() != null) {
            int startPosition = BleManager.getInstance(this.mActivity).getmConnBleDevice().getStartPosition();
            refreshStartBitUI(startPosition);
            int stopPosition = BleManager.getInstance(this.mActivity).getmConnBleDevice().getStopPosition();
            if (stopPosition <= 0) {
                stopPosition = this.start_stop_position_diff + startPosition;
            }
            refreshEndBitUI(stopPosition);
            refreshSmoothUI(BleManager.getInstance(this.mActivity).getmConnBleDevice().getSmoothness());
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        if (this.mPositionEnd != i) {
            this.mPositionEnd = i;
            this.txt_position_max.setText(this.mPositionEnd + "");
            double width = this.progress_position.getWidth();
            int i2 = (int) (width - ((width / 100.0d) * i));
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, i2, 0);
            this.layout_position_max.setLayoutParams(layoutParams);
        }
    }

    private void refreshSmoothUI(int i) {
        ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.dialog.PlayModeSettingDialog.4
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                PlayModeSettingDialog.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
            }
        });
        ofInt.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartBitUI(int i) {
        if (this.mPositionStart != i) {
            this.mPositionStart = i;
            this.txt_position_min.setText(this.mPositionStart + "");
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_min.getLayoutParams();
            layoutParams.setMargins((int) ((this.progress_position.getWidth() / 100.0d) * i), 0, 0, 0);
            this.layout_position_min.setLayoutParams(layoutParams);
        }
    }

    private void setListener() {
        this.img_back.setOnClickListener(this);
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.dialog.PlayModeSettingDialog.3
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + PlayModeSettingDialog.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + PlayModeSettingDialog.this.seek_smooth.getProgress());
                int progress = PlayModeSettingDialog.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                BleManager.getInstance(PlayModeSettingDialog.this.mActivity).writeSmooth(i / 100);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.dialog.PlayModeSettingDialog.3.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        PlayModeSettingDialog.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
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
        LogUtil.e("显示onAttachedToWindow" + this.progress_position.getWidth());
        onStatusListener onstatuslistener = this.mOnStatusListener;
        if (onstatuslistener != null) {
            onstatuslistener.onShow();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.img_back) {
            dismiss();
        }
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_play_mode_setting);
        Window window = getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (this.mActivity.getWindowManager().getDefaultDisplay().getWidth() * 0.9d);
            window.setAttributes(attributes);
        }
        this.img_back = (ImageView) findViewById(R.id.img_back);
        this.dialog_confirm_layout_content = (LinearLayout) findViewById(R.id.dialog_confirm_layout_content);
        this.layout_position = (ConstraintLayout) findViewById(R.id.layout_position);
        this.layout_smooth = (ConstraintLayout) findViewById(R.id.layout_smooth);
        this.layout_slide_speed = (LinearLayout) findViewById(R.id.layout_slide_speed);
        this.progress_position = (ProgressBar) findViewById(R.id.progress_position);
        this.layout_position_max = (ConstraintLayout) findViewById(R.id.layout_position_max);
        this.layout_position_min = (ConstraintLayout) findViewById(R.id.layout_position_min);
        this.txt_position_max = (TextView) findViewById(R.id.txt_position_max);
        this.txt_position_min = (TextView) findViewById(R.id.txt_position_min);
        this.seek_smooth = (CustomSeekbar) findViewById(R.id.seek_smooth);
        this.txt_slide_speed = (TextView) findViewById(R.id.txt_slide_speed);
        this.progress_slide_speed = (SettingProgressView) findViewById(R.id.progress_slide_speed);
        boolean z = this.mSlideShow;
        LinearLayout linearLayout = this.layout_slide_speed;
        int i = 0;
        if (z) {
            linearLayout.setVisibility(0);
        } else {
            linearLayout.setVisibility(8);
        }
        boolean z2 = this.mSMSUIShow;
        ConstraintLayout constraintLayout = this.layout_position;
        if (z2) {
            constraintLayout.setVisibility(0);
            this.layout_smooth.setVisibility(0);
        } else {
            constraintLayout.setVisibility(8);
            this.layout_smooth.setVisibility(8);
        }
        this.progress_slide_speed.setmItemCount(Constans.Slide_Speed_Arr.length);
        int intValue = ((Integer) SharedPreferencesUtil.get(this.mActivity, Constans.Shard_Slide_Speed_Position, -1)).intValue();
        if (intValue == -1 && MyApplication.getApplication().getmProductModel() != null) {
            while (true) {
                if (i >= Constans.Slide_Speed_Arr.length) {
                    break;
                }
                if (Constans.Slide_Speed_Arr[i] == MyApplication.getApplication().getmProductModel().getSlide()) {
                    intValue = i;
                    break;
                }
                i++;
            }
        }
        this.progress_slide_speed.setCurPosition(intValue);
        this.progress_slide_speed.setOnProgressChangeListener(new SettingProgressView.OnProgressChangeListener() { // from class: com.hismith.hismithapp.dialog.PlayModeSettingDialog.2
            @Override // com.hismith.hismithapp.view.SettingProgressView.OnProgressChangeListener
            public void onClick(int i2) {
                LogUtil.e("点击节点：" + i2);
                if (Constans.Slide_Speed_Arr.length <= i2 || i2 < 0) {
                    return;
                }
                SharedPreferencesUtil.put(PlayModeSettingDialog.this.mActivity, Constans.Shard_Slide_Speed_Position, Integer.valueOf(i2));
                PlayModeSettingDialog.this.mActivity.sendBroadcast(new Intent(Constans.RECEIVER_SLIDE_SPEED_ACTION));
            }

            @Override // com.hismith.hismithapp.view.SettingProgressView.OnProgressChangeListener
            public void onProgressChange(int i2) {
                LogUtil.e("节点：" + i2);
            }
        });
        LanguageUtil.setText(this.mActivity, this.txt_slide_speed, "speedSensitivity");
        setCanceledOnTouchOutside(true);
        setListener();
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        LogUtil.e("消失onDetachedFromWindow");
        onStatusListener onstatuslistener = this.mOnStatusListener;
        if (onstatuslistener != null) {
            onstatuslistener.onDismiss();
        }
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        if (z) {
            LogUtil.e("显示onWindowFocusChanged " + this.progress_position.getWidth());
            initBleData();
        }
    }

    @Override // android.app.Dialog
    public void show() {
        super.show();
        LogUtil.e("显示");
    }
}
