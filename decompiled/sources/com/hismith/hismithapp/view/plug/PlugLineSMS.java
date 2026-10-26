package com.hismith.hismithapp.view.plug;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CustomSeekbar;

/* loaded from: classes3.dex */
public class PlugLineSMS {
    private View baseView;
    private float downX;
    private float endDownX;
    private ImageView img_run_mode;
    private FrameLayout layout_position_max;
    private FrameLayout layout_position_min;
    private FrameLayout layout_speed_max;
    private Context mContext;
    private ProgressBar progress_position;
    private ProgressBar progress_speed;
    private CustomSeekbar seek_smooth;
    private float startDownX;
    private TextView txt_position;
    private TextView txt_position_label;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private TextView txt_smooth;
    private TextView txt_smooth_label;
    private TextView txt_speed;
    private TextView txt_speed_label;
    private TextView txt_speed_max;
    private int start_stop_position_diff = Constans.start_stop_position_diff;
    private int mSpeed = 0;
    private int mSafeMax = 100;
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugLineSMS.1
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlugLineSMS.this.downX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlugLineSMS.this.downX;
            int right = PlugLineSMS.this.progress_speed.getRight() + (PlugLineSMS.this.layout_speed_max.getWidth() / 2);
            LogUtil.e("getRight():" + PlugLineSMS.this.layout_speed_max.getRight() + ", width:" + PlugLineSMS.this.progress_speed.getRight() + ", " + PlugLineSMS.this.layout_speed_max.getWidth());
            int right2 = (int) (right - (PlugLineSMS.this.layout_speed_max.getRight() + x));
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > PlugLineSMS.this.progress_speed.getWidth()) {
                right2 = PlugLineSMS.this.progress_speed.getWidth();
            }
            LogUtil.e("safe right:" + right2);
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) PlugLineSMS.this.layout_speed_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            PlugLineSMS.this.layout_speed_max.setLayoutParams(layoutParams);
            int width = (int) (100.0d - ((right2 * 100.0d) / PlugLineSMS.this.progress_speed.getWidth()));
            PlugLineSMS.this.mSafeMax = width;
            PlugLineSMS.this.txt_speed_max.setText(width + "");
            if (width < 0 || width >= PlugLineSMS.this.mSpeed) {
                return true;
            }
            BleManager.getInstance(PlugLineSMS.this.mContext).writeSpeed(width);
            PlugLineSMS.this.mSpeed = width;
            PlugLineSMS.this.progress_speed.setProgress(width);
            PlugLineSMS.this.txt_speed.setText(width + "");
            return true;
        }
    };
    private int mPositionEnd = 100;
    private View.OnTouchListener endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugLineSMS.2
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlugLineSMS.this.endDownX = motionEvent.getX();
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlugLineSMS.this.endDownX;
            int right = PlugLineSMS.this.progress_position.getRight() + (PlugLineSMS.this.layout_position_max.getWidth() / 2);
            LogUtil.e("getRight:" + PlugLineSMS.this.layout_position_max.getRight());
            int right2 = (int) (right - (PlugLineSMS.this.layout_position_max.getRight() + x));
            LogUtil.e("onTouch moveX:" + x + ", right():" + right2);
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > PlugLineSMS.this.progress_position.getWidth()) {
                right2 = PlugLineSMS.this.progress_position.getWidth();
            }
            int width = (int) (100.0d - ((right2 * 100.0d) / PlugLineSMS.this.progress_position.getWidth()));
            if (width <= PlugLineSMS.this.mPositionStart + PlugLineSMS.this.start_stop_position_diff) {
                int i = PlugLineSMS.this.mPositionStart + PlugLineSMS.this.start_stop_position_diff;
                width = i > 100 ? 100 : i;
                right2 = (int) (PlugLineSMS.this.progress_position.getWidth() - ((PlugLineSMS.this.progress_position.getWidth() / 100.0d) * width));
            }
            int i2 = width <= 100 ? width < 0 ? 0 : width : 100;
            RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) PlugLineSMS.this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            PlugLineSMS.this.layout_position_max.setLayoutParams(layoutParams);
            PlugLineSMS.this.mPositionEnd = i2;
            LogUtil.e("结束位置：" + PlugLineSMS.this.mPositionEnd + ", right:" + right2);
            PlugLineSMS.this.txt_position_max.setText(i2 + "");
            BleManager.getInstance(PlugLineSMS.this.mContext).writeStopPosition(PlugLineSMS.this.mPositionEnd);
            return true;
        }
    };
    private int mPositionStart = 0;
    private View.OnTouchListener startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.view.plug.PlugLineSMS.3
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlugLineSMS.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlugLineSMS.this.startDownX;
            int width = PlugLineSMS.this.progress_position.getWidth();
            int left = (int) (PlugLineSMS.this.layout_position_min.getLeft() + x);
            LogUtil.e("onTouch moveX:" + x + ", left():" + left);
            if (left < 0) {
                left = 0;
            } else if (left > width) {
                left = width;
            }
            double d = width;
            int i = (int) ((left * 100.0d) / d);
            if (i >= PlugLineSMS.this.mPositionEnd - PlugLineSMS.this.start_stop_position_diff) {
                i = PlugLineSMS.this.mPositionEnd - PlugLineSMS.this.start_stop_position_diff;
                if (i < 0) {
                    i = 0;
                }
                left = (int) ((d / 100.0d) * i);
            }
            if (i > 100) {
                i = 100;
            } else if (i < 0) {
                i = 0;
            }
            RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) PlugLineSMS.this.layout_position_min.getLayoutParams();
            layoutParams.setMargins(left, 0, 0, 0);
            PlugLineSMS.this.layout_position_min.setLayoutParams(layoutParams);
            PlugLineSMS.this.mPositionStart = i;
            LogUtil.e("开始位置：" + PlugLineSMS.this.mPositionStart + ", left:" + left);
            PlugLineSMS.this.txt_position_min.setText(i + "");
            BleManager.getInstance(PlugLineSMS.this.mContext).writeStartPosition(PlugLineSMS.this.mPositionStart);
            return true;
        }
    };

    public PlugLineSMS(Context context) {
        this.mContext = context;
        this.baseView = LayoutInflater.from(context).inflate(R.layout.plug_line_sms, (ViewGroup) null, false);
        initView();
        init();
        setListener();
    }

    private void init() {
    }

    private void initView() {
        this.img_run_mode = (ImageView) this.baseView.findViewById(R.id.img_run_mode);
        this.txt_speed_label = (TextView) this.baseView.findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) this.baseView.findViewById(R.id.txt_speed);
        this.progress_speed = (ProgressBar) this.baseView.findViewById(R.id.progress_speed);
        this.layout_speed_max = (FrameLayout) this.baseView.findViewById(R.id.layout_speed_max);
        this.txt_speed_max = (TextView) this.baseView.findViewById(R.id.txt_speed_max);
        this.txt_position_label = (TextView) this.baseView.findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) this.baseView.findViewById(R.id.txt_position);
        this.txt_position_max = (TextView) this.baseView.findViewById(R.id.txt_position_max);
        this.txt_position_min = (TextView) this.baseView.findViewById(R.id.txt_position_min);
        this.progress_position = (ProgressBar) this.baseView.findViewById(R.id.progress_position);
        this.layout_position_max = (FrameLayout) this.baseView.findViewById(R.id.layout_position_max);
        this.layout_position_min = (FrameLayout) this.baseView.findViewById(R.id.layout_position_min);
        this.txt_smooth_label = (TextView) this.baseView.findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) this.baseView.findViewById(R.id.txt_smooth);
        this.seek_smooth = (CustomSeekbar) this.baseView.findViewById(R.id.seek_smooth);
    }

    private void setListener() {
        this.layout_speed_max.setOnTouchListener(this.safeOnTouch);
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
    }

    public View getBaseView() {
        return this.baseView;
    }
}
