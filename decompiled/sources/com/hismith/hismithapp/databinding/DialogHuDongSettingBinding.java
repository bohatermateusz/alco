package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomSeekbar;

/* loaded from: classes3.dex */
public final class DialogHuDongSettingBinding implements ViewBinding {
    public final LinearLayout dialogConfirmLayoutContent;
    public final ImageView imgBack;
    public final ImageView imgModeSwitch;
    public final ConstraintLayout layoutMode;
    public final FrameLayout layoutSpeedMax;
    public final ConstraintLayout layoutSpeedSafe;
    public final FrameLayout layoutVibrateMax;
    public final RelativeLayout layoutVibrateSafe;
    public final ProgressBar progressSpeed;
    public final ProgressBar progressVibrate;
    private final LinearLayout rootView;
    public final CustomSeekbar seekSmooth;
    public final TextView txtModeLabel;
    public final TextView txtNoConnected;
    public final AppCompatTextView txtSpeedMax;
    public final AppCompatTextView txtVibrateMax;

    private DialogHuDongSettingBinding(LinearLayout linearLayout, LinearLayout linearLayout2, ImageView imageView, ImageView imageView2, ConstraintLayout constraintLayout, FrameLayout frameLayout, ConstraintLayout constraintLayout2, FrameLayout frameLayout2, RelativeLayout relativeLayout, ProgressBar progressBar, ProgressBar progressBar2, CustomSeekbar customSeekbar, TextView textView, TextView textView2, AppCompatTextView appCompatTextView, AppCompatTextView appCompatTextView2) {
        this.rootView = linearLayout;
        this.dialogConfirmLayoutContent = linearLayout2;
        this.imgBack = imageView;
        this.imgModeSwitch = imageView2;
        this.layoutMode = constraintLayout;
        this.layoutSpeedMax = frameLayout;
        this.layoutSpeedSafe = constraintLayout2;
        this.layoutVibrateMax = frameLayout2;
        this.layoutVibrateSafe = relativeLayout;
        this.progressSpeed = progressBar;
        this.progressVibrate = progressBar2;
        this.seekSmooth = customSeekbar;
        this.txtModeLabel = textView;
        this.txtNoConnected = textView2;
        this.txtSpeedMax = appCompatTextView;
        this.txtVibrateMax = appCompatTextView2;
    }

    public static DialogHuDongSettingBinding bind(View view) {
        int i = R.id.dialog_confirm_layout_content;
        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
        if (linearLayout != null) {
            i = R.id.img_back;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_mode_switch;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.layout_mode;
                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                    if (constraintLayout != null) {
                        i = R.id.layout_speed_max;
                        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                        if (frameLayout != null) {
                            i = R.id.layout_speed_safe;
                            ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout2 != null) {
                                i = R.id.layout_vibrate_max;
                                FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                if (frameLayout2 != null) {
                                    i = R.id.layout_vibrate_safe;
                                    RelativeLayout relativeLayout = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                                    if (relativeLayout != null) {
                                        i = R.id.progress_speed;
                                        ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                        if (progressBar != null) {
                                            i = R.id.progress_vibrate;
                                            ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                            if (progressBar2 != null) {
                                                i = R.id.seek_smooth;
                                                CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
                                                if (customSeekbar != null) {
                                                    i = R.id.txt_mode_label;
                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView != null) {
                                                        i = R.id.txt_no_connected;
                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView2 != null) {
                                                            i = R.id.txt_speed_max;
                                                            AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                            if (appCompatTextView != null) {
                                                                i = R.id.txt_vibrate_max;
                                                                AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                if (appCompatTextView2 != null) {
                                                                    return new DialogHuDongSettingBinding((LinearLayout) view, linearLayout, imageView, imageView2, constraintLayout, frameLayout, constraintLayout2, frameLayout2, relativeLayout, progressBar, progressBar2, customSeekbar, textView, textView2, appCompatTextView, appCompatTextView2);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogHuDongSettingBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogHuDongSettingBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_hu_dong_setting, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public LinearLayout getRoot() {
        return this.rootView;
    }
}
