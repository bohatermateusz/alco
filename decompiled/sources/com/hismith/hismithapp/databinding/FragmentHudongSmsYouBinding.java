package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public final class FragmentHudongSmsYouBinding implements ViewBinding {
    public final ImageView imgPosition;
    public final ImageView imgSpeedMode;
    public final ImageView imgSpeedModeBg;
    public final ImageView imgVibrateMode;
    public final ImageView imgVibrateModeBg;
    public final ConstraintLayout layoutPlay;
    public final ConstraintLayout layoutRoot;
    public final ConstraintLayout layoutSpeedMode;
    public final ConstraintLayout layoutTouchPosition;
    public final ConstraintLayout layoutVibrateMode;
    public final View line1;
    public final PlayView_Four playView;
    public final ProgressBar progressPosition;
    public final VerticalProgressBar progressSpeed;
    private final ConstraintLayout rootView;
    public final CustomWebView webDevice;

    private FragmentHudongSmsYouBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, ConstraintLayout constraintLayout4, ConstraintLayout constraintLayout5, ConstraintLayout constraintLayout6, View view, PlayView_Four playView_Four, ProgressBar progressBar, VerticalProgressBar verticalProgressBar, CustomWebView customWebView) {
        this.rootView = constraintLayout;
        this.imgPosition = imageView;
        this.imgSpeedMode = imageView2;
        this.imgSpeedModeBg = imageView3;
        this.imgVibrateMode = imageView4;
        this.imgVibrateModeBg = imageView5;
        this.layoutPlay = constraintLayout2;
        this.layoutRoot = constraintLayout3;
        this.layoutSpeedMode = constraintLayout4;
        this.layoutTouchPosition = constraintLayout5;
        this.layoutVibrateMode = constraintLayout6;
        this.line1 = view;
        this.playView = playView_Four;
        this.progressPosition = progressBar;
        this.progressSpeed = verticalProgressBar;
        this.webDevice = customWebView;
    }

    public static FragmentHudongSmsYouBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_position;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_speed_mode;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_speed_mode_bg;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.img_vibrate_mode;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.img_vibrate_mode_bg;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.layout_play;
                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout != null) {
                                ConstraintLayout constraintLayout2 = (ConstraintLayout) view;
                                i = R.id.layout_speed_mode;
                                ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                if (constraintLayout3 != null) {
                                    i = R.id.layout_touch_position;
                                    ConstraintLayout constraintLayout4 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                    if (constraintLayout4 != null) {
                                        i = R.id.layout_vibrate_mode;
                                        ConstraintLayout constraintLayout5 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                        if (constraintLayout5 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line1))) != null) {
                                            i = R.id.playView;
                                            PlayView_Four playView_Four = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                            if (playView_Four != null) {
                                                i = R.id.progress_position;
                                                ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                if (progressBar != null) {
                                                    i = R.id.progress_speed;
                                                    VerticalProgressBar verticalProgressBar = (VerticalProgressBar) ViewBindings.findChildViewById(view, i);
                                                    if (verticalProgressBar != null) {
                                                        i = R.id.web_device;
                                                        CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                                        if (customWebView != null) {
                                                            return new FragmentHudongSmsYouBinding(constraintLayout2, imageView, imageView2, imageView3, imageView4, imageView5, constraintLayout, constraintLayout2, constraintLayout3, constraintLayout4, constraintLayout5, findChildViewById, playView_Four, progressBar, verticalProgressBar, customWebView);
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

    public static FragmentHudongSmsYouBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentHudongSmsYouBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sms_you, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public ConstraintLayout getRoot() {
        return this.rootView;
    }
}
