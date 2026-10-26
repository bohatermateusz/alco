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
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.yhao.floatwindow.FloatWindow;
import com.yhao.floatwindow.ViewStateListener;

/* loaded from: classes3.dex */
public class FloatSM extends FloatBase {
    private int lastY;
    private Context mContext;
    private ImageView mFloatImgBg;
    private ImageView mFloatImgHeart;
    private ImageView mFloatImgMode;
    private FrameLayout mFloatLayout;
    private TextView mFloatTxtSpeed;
    private View mFloatView;
    private View mFloatViewMode;
    private String mTag;
    private ImageView popup_float_mode_bg;

    public FloatSM(Context context, String str, View.OnClickListener onClickListener) {
        super(context, str, onClickListener);
        this.lastY = 0;
        this.mContext = context;
        this.mTag = str;
        initView(onClickListener);
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    void initView(View.OnClickListener onClickListener) {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.popup_floating_sm, (ViewGroup) null, false);
        this.mFloatView = inflate;
        this.mFloatViewMode = inflate.findViewById(R.id.popup_view_mode);
        this.mFloatImgBg = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_bg);
        this.mFloatLayout = (FrameLayout) this.mFloatView.findViewById(R.id.popup_float_layout);
        this.popup_float_mode_bg = (ImageView) this.mFloatView.findViewById(R.id.popup_float_mode_bg);
        this.mFloatImgMode = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_mode);
        this.mFloatTxtSpeed = (TextView) this.mFloatView.findViewById(R.id.popup_float_txt_speed);
        this.mFloatImgHeart = (ImageView) this.mFloatView.findViewById(R.id.popup_float_img_heart);
        this.popup_float_mode_bg.setVisibility(4);
        this.mFloatImgMode.setVisibility(4);
        this.mFloatImgHeart.setVisibility(4);
        this.mFloatView.setOnClickListener(onClickListener);
        if (BleManager.getInstance(this.mContext).getmConnBleDevice() == null) {
            refreshFloatSpeedUI(0);
            return;
        }
        int mode = BleManager.getInstance(this.mContext).getmConnBleDevice().getMode();
        Context context = this.mContext;
        if (mode > 0) {
            refreshFloatModeUI(BleManager.getInstance(context).getmConnBleDevice().getMode());
        } else {
            refreshFloatSpeedUI(BleManager.getInstance(context).getmConnBleDevice().getSpeed());
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatClimaxUI(int i) {
        ImageView imageView = this.mFloatImgHeart;
        if (i == 0) {
            AnimUtil.startHeart(imageView);
        } else {
            AnimUtil.stopHeart(imageView);
        }
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatModeUI(int i) {
        AnimUtil.stopHeart(this.mFloatImgHeart);
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
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatSmooth(int i) {
    }

    @Override // com.hismith.hismithapp.floating.FloatBase
    public void refreshFloatSpeedUI(int i) {
        AnimUtil.stopHeart(this.mFloatImgHeart);
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
            FloatWindow.with(this.mContext.getApplicationContext()).setView(this.mFloatView).setTag(this.mTag).setWidth(dip2px).setHeight(dip2px2).setX(screenSize.x - (dip2px + dip2px3)).setFilter(false, ControlSponsorActivity2.class, ControlAcceptActivity2.class).setY(1, 0.3f).setMoveType(3, dip2px3, dip2px3).setMoveStyle(500L, new BounceInterpolator()).setDesktopShow(true).setViewStateListener(new ViewStateListener() { // from class: com.hismith.hismithapp.floating.FloatSM.1
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
                    Point screenSize2 = MeasureUtil.getScreenSize(FloatSM.this.mContext);
                    if (FloatWindow.get(FloatSM.this.mTag) != null) {
                        if (i2 < 0) {
                            FloatWindow.get(FloatSM.this.mTag).updateY(0);
                        } else if (i2 > ((screenSize2.y - dip2px2) - MeasureUtil.getStatusBarHeight(FloatSM.this.mContext)) - dip2px3) {
                            FloatWindow.get(FloatSM.this.mTag).updateY(FloatSM.this.lastY);
                            return;
                        }
                        FloatSM.this.lastY = i2;
                    }
                }

                @Override // com.yhao.floatwindow.ViewStateListener
                public void onShow() {
                }
            }).build();
            try {
                FloatWindow.get(this.mTag).show();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
