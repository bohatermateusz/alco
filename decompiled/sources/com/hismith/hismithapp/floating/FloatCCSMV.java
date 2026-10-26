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
public class FloatCCSMV extends FloatBase {
    private int lastY;
    private Context mContext;
    private ImageView mFloatImgBg;
    private ImageView mFloatImgMode;
    private ImageView mFloatImgVibrate;
    private FrameLayout mFloatLayout;
    private TextView mFloatTxtSpeed;
    private TextView mFloatTxtVibrate;
    private View mFloatView;
    private View mFloatViewMode;
    private View mFloatViewVibrate;
    private String mTag;
    private ImageView popup_float_mode_bg;
    private ImageView popup_float_vibrate_bg;

    public FloatCCSMV(Context context, String str, View.OnClickListener onClickListener) {
        super(context, str, onClickListener);
        this.lastY = 0;
        this.mContext = context;
        this.mTag = str;
        initView(onClickListener);
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    void initView(View.OnClickListener onClickListener) {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.popup_floating_cc_smv, (ViewGroup) null, false);
        this.mFloatView = inflate;
        this.mFloatViewMode = inflate.findViewById(R.id.popup_view_mode);
        this.mFloatViewVibrate = this.mFloatView.findViewById(R.id.popup_view_vibrate);
        this.mFloatImgBg = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_bg);
        this.mFloatLayout = (FrameLayout) this.mFloatView.findViewById(R.id.popup_float_layout);
        this.popup_float_mode_bg = (ImageView) this.mFloatView.findViewById(R.id.popup_float_mode_bg);
        this.mFloatImgMode = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_mode);
        this.popup_float_vibrate_bg = (ImageView) this.mFloatView.findViewById(R.id.popup_float_vibrate_bg);
        this.mFloatImgVibrate = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_vibrate);
        this.mFloatTxtSpeed = (TextView) this.mFloatView.findViewById(R.id.popup_float_txt_speed);
        this.mFloatTxtVibrate = (TextView) this.mFloatView.findViewById(R.id.popup_float_txt_vibrate);
        this.popup_float_mode_bg.setVisibility(4);
        this.mFloatImgMode.setVisibility(4);
        this.popup_float_vibrate_bg.setVisibility(4);
        this.mFloatImgVibrate.setVisibility(4);
        this.mFloatView.setOnClickListener(onClickListener);
        if (BleManager.getInstance(this.mContext).getmConnBleDevice() == null) {
            refreshFloatSpeedUI(0);
            refreshFloatVibrateUI(0);
            return;
        }
        int mode = BleManager.getInstance(this.mContext).getmConnBleDevice().getMode();
        Context context = this.mContext;
        if (mode > 0) {
            refreshFloatModeUI(BleManager.getInstance(context).getmConnBleDevice().getMode());
        } else {
            refreshFloatSpeedUI(BleManager.getInstance(context).getmConnBleDevice().getSpeed());
        }
        int vibrationMode = BleManager.getInstance(this.mContext).getmConnBleDevice().getVibrationMode();
        Context context2 = this.mContext;
        if (vibrationMode > 0) {
            refreshFloatModeUI(BleManager.getInstance(context2).getmConnBleDevice().getVibrationMode());
        } else {
            refreshFloatVibrateUI(BleManager.getInstance(context2).getmConnBleDevice().getVibration());
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatClimaxUI(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatModeUI(int i) {
        TextView textView = this.mFloatTxtSpeed;
        if (i <= 0) {
            if (textView != null) {
                textView.setVisibility(0);
            }
            ImageView imageView = this.popup_float_mode_bg;
            if (imageView != null) {
                imageView.setVisibility(4);
            }
            ImageView imageView2 = this.mFloatImgMode;
            if (imageView2 != null) {
                imageView2.setVisibility(4);
            }
            View view = this.mFloatViewMode;
            if (view != null) {
                ((GradientDrawable) view.getBackground()).setColor(Color.parseColor("#8C8C8C"));
                return;
            }
            return;
        }
        if (textView != null) {
            textView.setVisibility(4);
        }
        View view2 = this.mFloatViewMode;
        if (view2 != null) {
            ((GradientDrawable) view2.getBackground()).setColor(this.mContext.getResources().getColor(R.color.app_style_color));
        }
        ImageView imageView3 = this.popup_float_mode_bg;
        if (imageView3 != null) {
            imageView3.setVisibility(0);
        }
        ImageView imageView4 = this.mFloatImgMode;
        if (imageView4 != null) {
            imageView4.setVisibility(0);
            if (ProductUtil.hasMode(0, i)) {
                Glide.with(MyApplication.getApplication()).load(ProductUtil.getProductMode(0)[i - 1].getmIcon()).placeholder(R.mipmap.ic_iplay_def_button).into(this.mFloatImgMode);
            }
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatMusicUI(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatPosition(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatReversal(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatRunMode(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatShutdown() {
        refreshFloatSpeedUI(0);
        refreshFloatVibrateUI(0);
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatSmooth(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatSpeedUI(int i) {
        ImageView imageView = this.popup_float_mode_bg;
        if (imageView != null) {
            imageView.setVisibility(4);
        }
        ImageView imageView2 = this.mFloatImgMode;
        if (imageView2 != null) {
            imageView2.setVisibility(4);
        }
        View view = this.mFloatViewMode;
        if (view != null) {
            ((GradientDrawable) view.getBackground()).setColor(Color.parseColor("#8C8C8C"));
        }
        TextView textView = this.mFloatTxtSpeed;
        if (textView != null) {
            textView.setVisibility(0);
            this.mFloatTxtSpeed.setText(i + "");
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatStartPosition(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatStopPosition(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatVibrateModeUI(int i) {
        TextView textView = this.mFloatTxtVibrate;
        if (i <= 0) {
            if (textView != null) {
                textView.setVisibility(0);
            }
            ImageView imageView = this.popup_float_vibrate_bg;
            if (imageView != null) {
                imageView.setVisibility(4);
            }
            ImageView imageView2 = this.mFloatImgVibrate;
            if (imageView2 != null) {
                imageView2.setVisibility(4);
            }
            View view = this.mFloatViewVibrate;
            if (view != null) {
                ((GradientDrawable) view.getBackground()).setColor(Color.parseColor("#8C8C8C"));
                return;
            }
            return;
        }
        if (textView != null) {
            textView.setVisibility(4);
        }
        View view2 = this.mFloatViewVibrate;
        if (view2 != null) {
            ((GradientDrawable) view2.getBackground()).setColor(this.mContext.getResources().getColor(R.color.app_style_color));
        }
        ImageView imageView3 = this.popup_float_vibrate_bg;
        if (imageView3 != null) {
            imageView3.setVisibility(0);
        }
        ImageView imageView4 = this.mFloatImgVibrate;
        if (imageView4 != null) {
            imageView4.setVisibility(0);
            if (ProductUtil.hasMode(1, i)) {
                Glide.with(MyApplication.getApplication()).load(ProductUtil.getProductMode(1)[i - 1].getmIcon()).placeholder(R.mipmap.ic_iplay_def_button).into(this.mFloatImgVibrate);
            }
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatVibrateUI(int i) {
        ImageView imageView = this.popup_float_vibrate_bg;
        if (imageView != null) {
            imageView.setVisibility(4);
        }
        ImageView imageView2 = this.mFloatImgVibrate;
        if (imageView2 != null) {
            imageView2.setVisibility(4);
        }
        View view = this.mFloatViewVibrate;
        if (view != null) {
            ((GradientDrawable) view.getBackground()).setColor(Color.parseColor("#8C8C8C"));
        }
        TextView textView = this.mFloatTxtVibrate;
        if (textView != null) {
            textView.setVisibility(0);
            this.mFloatTxtVibrate.setText(i + "");
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void show() {
        if (FloatWindow.get(this.mTag) == null || !FloatWindow.get(this.mTag).isShowing()) {
            Point screenSize = MeasureUtil.getScreenSize(this.mContext);
            int dip2px = MeasureUtil.dip2px(this.mContext, 75.0f);
            final int dip2px2 = MeasureUtil.dip2px(this.mContext, 150.0f);
            final int dip2px3 = MeasureUtil.dip2px(this.mContext, 6.0f);
            FloatWindow.with(this.mContext.getApplicationContext()).setView(this.mFloatView).setTag(this.mTag).setWidth(dip2px).setHeight(dip2px2).setX(screenSize.x - (dip2px + dip2px3)).setFilter(false, ControlSponsorActivity2.class, ControlAcceptActivity2.class).setY(1, 0.3f).setMoveType(3, dip2px3, dip2px3).setMoveStyle(500L, new BounceInterpolator()).setDesktopShow(true).setViewStateListener(new ViewStateListener() { // from class: com.hismith.hismithapp.floating.FloatCCSMV.1
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
                    Point screenSize2 = MeasureUtil.getScreenSize(FloatCCSMV.this.mContext);
                    if (FloatWindow.get(FloatCCSMV.this.mTag) != null) {
                        if (i2 < 0) {
                            FloatWindow.get(FloatCCSMV.this.mTag).updateY(0);
                        } else if (i2 > ((screenSize2.y - dip2px2) - MeasureUtil.getStatusBarHeight(FloatCCSMV.this.mContext)) - dip2px3) {
                            FloatWindow.get(FloatCCSMV.this.mTag).updateY(FloatCCSMV.this.lastY);
                            return;
                        }
                        FloatCCSMV.this.lastY = i2;
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
