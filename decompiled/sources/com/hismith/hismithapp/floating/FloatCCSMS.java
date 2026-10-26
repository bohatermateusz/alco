package com.hismith.hismithapp.floating;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Point;
import android.graphics.drawable.GradientDrawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.BounceInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.yhao.floatwindow.FloatWindow;
import com.yhao.floatwindow.ViewStateListener;

/* loaded from: classes3.dex */
public class FloatCCSMS extends FloatBase {
    private int lastY;
    private Context mContext;
    private ImageView mFloatImgBg;
    private ImageView mFloatImgMode;
    private FrameLayout mFloatLayout;
    private TextView mFloatTxtAuto;
    private TextView mFloatTxtSpeed;
    private View mFloatView;
    private View mFloatViewMode;
    private String mTag;
    private ImageView popup_float_mode_bg;

    public FloatCCSMS(Context context, String str, View.OnClickListener onClickListener) {
        super(context, str, onClickListener);
        this.lastY = 0;
        this.mContext = context;
        this.mTag = str;
        initView(onClickListener);
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    void initView(View.OnClickListener onClickListener) {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.popup_floating_cc_sms, (ViewGroup) null, false);
        this.mFloatView = inflate;
        this.mFloatTxtAuto = (TextView) inflate.findViewById(R.id.popup_txt_auto);
        this.mFloatViewMode = this.mFloatView.findViewById(R.id.popup_view_mode);
        this.mFloatImgBg = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_bg);
        this.mFloatLayout = (FrameLayout) this.mFloatView.findViewById(R.id.popup_float_layout);
        this.mFloatImgMode = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_mode);
        this.popup_float_mode_bg = (ImageView) this.mFloatView.findViewById(R.id.popup_float_mode_bg);
        this.mFloatTxtSpeed = (TextView) this.mFloatView.findViewById(R.id.popup_float_txt_speed);
        this.popup_float_mode_bg.setVisibility(4);
        this.mFloatImgMode.setVisibility(4);
        this.mFloatView.setOnClickListener(onClickListener);
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatClimaxUI(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatModeUI(int i) {
        TextView textView = this.mFloatTxtSpeed;
        if (i <= 0) {
            textView.setVisibility(0);
            this.popup_float_mode_bg.setVisibility(4);
            this.mFloatImgMode.setVisibility(4);
            ((GradientDrawable) this.mFloatViewMode.getBackground()).setColor(Color.parseColor("#8C8C8C"));
            return;
        }
        textView.setVisibility(4);
        ((GradientDrawable) this.mFloatViewMode.getBackground()).setColor(this.mContext.getResources().getColor(R.color.app_style_color));
        this.popup_float_mode_bg.setVisibility(0);
        this.mFloatImgMode.setVisibility(0);
        if (ProductUtil.hasMode(0, i)) {
            Glide.with(MyApplication.getApplication()).load(ProductUtil.getProductMode(0)[i - 1].getmIcon()).placeholder(R.mipmap.ic_iplay_def_button).into(this.mFloatImgMode);
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatMusicUI(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatPosition(int i) {
        if (BleManager.getInstance(this.mContext).getmConnBleDevice() != null) {
            BleManager.getInstance(this.mContext).getmConnBleDevice().getRunMode();
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatReversal(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatRunMode(int i) {
        TextView textView = this.mFloatTxtAuto;
        if (i == 0) {
            textView.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
        } else {
            textView.setTextColor(Color.parseColor("#8C8C8C"));
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatShutdown() {
        refreshFloatSpeedUI(0);
        refreshFloatRunMode(0);
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatSmooth(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatSpeedUI(int i) {
        this.popup_float_mode_bg.setVisibility(4);
        this.mFloatImgMode.setVisibility(4);
        ((GradientDrawable) this.mFloatViewMode.getBackground()).setColor(Color.parseColor("#8C8C8C"));
        this.mFloatTxtSpeed.setVisibility(0);
        this.mFloatTxtSpeed.setText(i + "");
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatStartPosition(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatStopPosition(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatVibrateModeUI(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatVibrateUI(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void show() {
        if (FloatWindow.get(this.mTag) == null || !FloatWindow.get(this.mTag).isShowing()) {
            Point screenSize = MeasureUtil.getScreenSize(this.mContext);
            int dip2px = MeasureUtil.dip2px(this.mContext, 75.0f);
            final int dip2px2 = MeasureUtil.dip2px(this.mContext, 115.0f);
            final int dip2px3 = MeasureUtil.dip2px(this.mContext, 6.0f);
            FloatWindow.with(this.mContext.getApplicationContext()).setView(this.mFloatView).setTag(this.mTag).setWidth(dip2px).setHeight(dip2px2).setX(screenSize.x - (dip2px + dip2px3)).setFilter(false, ControlSponsorActivity2.class, ControlAcceptActivity2.class).setY(1, 0.3f).setMoveType(3, dip2px3, dip2px3).setMoveStyle(500L, new BounceInterpolator()).setDesktopShow(true).setViewStateListener(new ViewStateListener() { // from class: com.hismith.hismithapp.floating.FloatCCSMS.1
                @Override // com.yhao.floatwindow.ViewStateListener
                public void onBackToDesktop() {
                }

                @Override // com.yhao.floatwindow.ViewStateListener
                public void onDismiss() {
                }

                @Override // com.yhao.floatwindow.ViewStateListener
                public void onHide() {
                }

                @Override // com.yhao.floatwindow.ViewStateListener
                public void onMoveAnimEnd() {
                }

                @Override // com.yhao.floatwindow.ViewStateListener
                public void onMoveAnimStart() {
                }

                @Override // com.yhao.floatwindow.ViewStateListener
                public void onPositionUpdate(int i, int i2) {
                    Point screenSize2 = MeasureUtil.getScreenSize(FloatCCSMS.this.mContext);
                    if (FloatWindow.get(FloatCCSMS.this.mTag) != null) {
                        if (i2 < 0) {
                            FloatWindow.get(FloatCCSMS.this.mTag).updateY(0);
                        } else if (i2 > ((screenSize2.y - dip2px2) - MeasureUtil.getStatusBarHeight(FloatCCSMS.this.mContext)) - dip2px3) {
                            FloatWindow.get(FloatCCSMS.this.mTag).updateY(FloatCCSMS.this.lastY);
                            return;
                        }
                        FloatCCSMS.this.lastY = i2;
                    }
                }

                @Override // com.yhao.floatwindow.ViewStateListener
                public void onShow() {
                }
            }).build();
            FloatWindow.get(this.mTag).show();
        }
    }
}
