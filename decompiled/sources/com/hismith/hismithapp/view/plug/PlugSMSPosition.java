package com.hismith.hismithapp.view.plug;

import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class PlugSMSPosition {
    private View baseView;
    private float endDownX;
    private ConstraintLayout layout_position_max;
    private ConstraintLayout layout_position_min;
    private Context mContext;
    private ProgressBar progress_position;
    private float startDownX;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private int start_stop_position_diff = Constans.start_stop_position_diff;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.view.plug.PlugSMSPosition.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int intValue = ((Integer) message.obj).intValue();
            int i = message.what;
            if (i == 7) {
                PlugSMSPosition.this.refreshStartBitUI(intValue);
                return false;
            }
            if (i != 8) {
                return false;
            }
            PlugSMSPosition.this.refreshEndBitUI(intValue);
            return false;
        }
    });
    private int mPositionEnd = 100;
    private View.OnTouchListener endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugSMSPosition.4
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlugSMSPosition.this.endDownX = motionEvent.getX();
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlugSMSPosition.this.endDownX;
            int width = PlugSMSPosition.this.progress_position.getWidth();
            int right = (int) ((PlugSMSPosition.this.progress_position.getRight() + (PlugSMSPosition.this.layout_position_max.getWidth() / 2)) - (PlugSMSPosition.this.layout_position_max.getRight() + x));
            LogUtil.e("onTouch moveX:" + x + ", right():" + right);
            if (right < 0) {
                right = 0;
            } else if (right > width) {
                right = width;
            }
            double d = width;
            int i = (int) (100.0d - ((right * 100.0d) / d));
            if (i <= PlugSMSPosition.this.mPositionStart + PlugSMSPosition.this.start_stop_position_diff) {
                int i2 = PlugSMSPosition.this.mPositionStart + PlugSMSPosition.this.start_stop_position_diff;
                i = i2 > 100 ? 100 : i2;
                right = (int) (d - ((d / 100.0d) * i));
            }
            int i3 = i <= 100 ? i < 0 ? 0 : i : 100;
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlugSMSPosition.this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right, 0);
            PlugSMSPosition.this.layout_position_max.setLayoutParams(layoutParams);
            if (PlugSMSPosition.this.mPositionEnd == i3) {
                return true;
            }
            PlugSMSPosition.this.mPositionEnd = i3;
            LogUtil.e("结束位置：" + PlugSMSPosition.this.mPositionEnd + ", right:" + right);
            PlugSMSPosition.this.txt_position_max.setText(i3 + "");
            BleManager.getInstance(PlugSMSPosition.this.mContext).writeStopPosition(PlugSMSPosition.this.mPositionEnd);
            return true;
        }
    };
    private int mPositionStart = 0;
    private View.OnTouchListener startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugSMSPosition.5
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlugSMSPosition.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlugSMSPosition.this.startDownX;
            LogUtil.e("抖动：" + motionEvent.getX() + ", " + PlugSMSPosition.this.startDownX + " = " + x);
            int width = PlugSMSPosition.this.progress_position.getWidth();
            int left = (int) ((PlugSMSPosition.this.layout_position_min.getLeft() + x) - (PlugSMSPosition.this.progress_position.getLeft() - (PlugSMSPosition.this.layout_position_min.getWidth() / 2)));
            LogUtil.e("onTouch moveX:" + x + ", left():" + left);
            if (left < 0) {
                left = 0;
            } else if (left > width) {
                left = width;
            }
            double d = width;
            int i = (int) ((left * 100.0d) / d);
            if (i >= PlugSMSPosition.this.mPositionEnd - PlugSMSPosition.this.start_stop_position_diff) {
                int i2 = PlugSMSPosition.this.mPositionEnd - PlugSMSPosition.this.start_stop_position_diff;
                i = i2 < 0 ? 0 : i2;
                left = (int) ((d / 100.0d) * i);
            }
            if (i > 100) {
                i = 100;
            } else if (i < 0) {
                i = 0;
            }
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlugSMSPosition.this.layout_position_min.getLayoutParams();
            layoutParams.setMargins(left, 0, 0, 0);
            PlugSMSPosition.this.layout_position_min.setLayoutParams(layoutParams);
            if (PlugSMSPosition.this.mPositionStart == i) {
                return true;
            }
            PlugSMSPosition.this.mPositionStart = i;
            LogUtil.e("开始位置：" + PlugSMSPosition.this.mPositionStart + ", left:" + left);
            PlugSMSPosition.this.txt_position_min.setText(i + "");
            BleManager.getInstance(PlugSMSPosition.this.mContext).writeStartPosition(PlugSMSPosition.this.mPositionStart);
            return true;
        }
    };

    public PlugSMSPosition(Context context) {
        this.mContext = context;
        this.baseView = LayoutInflater.from(context).inflate(R.layout.plug_sms_position, (ViewGroup) null, false);
        initView();
        init();
        setListener();
    }

    private void init() {
    }

    private void initView() {
        this.progress_position = (ProgressBar) this.baseView.findViewById(R.id.progress_position);
        this.layout_position_max = (ConstraintLayout) this.baseView.findViewById(R.id.layout_position_max);
        this.txt_position_max = (TextView) this.baseView.findViewById(R.id.txt_position_max);
        this.layout_position_min = (ConstraintLayout) this.baseView.findViewById(R.id.layout_position_min);
        this.txt_position_min = (TextView) this.baseView.findViewById(R.id.txt_position_min);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        if (this.mPositionEnd != i) {
            this.mPositionEnd = i;
            this.txt_position_max.setText(this.mPositionEnd + "");
            double width = this.progress_position.getWidth();
            final int i2 = (int) (width - ((width / 100.0d) * i));
            this.layout_position_max.post(new Runnable() { // from class: com.hismith.hismithapp.view.plug.PlugSMSPosition.2
                @Override // java.lang.Runnable
                public void run() {
                    ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlugSMSPosition.this.layout_position_max.getLayoutParams();
                    layoutParams.setMargins(0, 0, i2, 0);
                    PlugSMSPosition.this.layout_position_max.setLayoutParams(layoutParams);
                }
            });
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartBitUI(int i) {
        if (this.mPositionStart != i) {
            this.mPositionStart = i;
            this.txt_position_min.setText(this.mPositionStart + "");
            final int width = (int) ((this.progress_position.getWidth() / 100.0d) * i);
            this.layout_position_min.post(new Runnable() { // from class: com.hismith.hismithapp.view.plug.PlugSMSPosition.3
                @Override // java.lang.Runnable
                public void run() {
                    ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlugSMSPosition.this.layout_position_min.getLayoutParams();
                    layoutParams.setMargins(width, 0, 0, 0);
                    PlugSMSPosition.this.layout_position_min.setLayoutParams(layoutParams);
                }
            });
        }
    }

    private void setListener() {
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
    }

    public View getBaseView() {
        return this.baseView;
    }

    public int getmPositionEnd() {
        return this.mPositionEnd;
    }

    public int getmPositionStart() {
        return this.mPositionStart;
    }

    public void setmPositionEnd(int i) {
        Message message = new Message();
        message.what = 8;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessageDelayed(message, 100L);
    }

    public void setmPositionStart(int i) {
        Message message = new Message();
        message.what = 7;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessageDelayed(message, 100L);
    }
}
