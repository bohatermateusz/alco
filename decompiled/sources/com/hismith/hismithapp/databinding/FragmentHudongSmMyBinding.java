package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomWebView;

/* loaded from: classes3.dex */
public final class FragmentHudongSmMyBinding implements ViewBinding {
    public final ImageView imgReversal;
    public final ImageView imgSpeedMode;
    public final ImageView imgSpeedModeBg;
    public final ImageView imgVibrateMode;
    public final ImageView imgVibrateModeBg;
    public final ConstraintLayout layoutRoot;
    public final ConstraintLayout layoutSpeedMode;
    public final ConstraintLayout layoutVibrateMode;
    public final View line1;
    private final ConstraintLayout rootView;
    public final CustomWebView webDevice;

    private FragmentHudongSmMyBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, ConstraintLayout constraintLayout4, View view, CustomWebView customWebView) {
        this.rootView = constraintLayout;
        this.imgReversal = imageView;
        this.imgSpeedMode = imageView2;
        this.imgSpeedModeBg = imageView3;
        this.imgVibrateMode = imageView4;
        this.imgVibrateModeBg = imageView5;
        this.layoutRoot = constraintLayout2;
        this.layoutSpeedMode = constraintLayout3;
        this.layoutVibrateMode = constraintLayout4;
        this.line1 = view;
        this.webDevice = customWebView;
    }

    public static FragmentHudongSmMyBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_reversal;
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
                            ConstraintLayout constraintLayout = (ConstraintLayout) view;
                            i = R.id.layout_speed_mode;
                            ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout2 != null) {
                                i = R.id.layout_vibrate_mode;
                                ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                if (constraintLayout3 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line1))) != null) {
                                    i = R.id.web_device;
                                    CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                    if (customWebView != null) {
                                        return new FragmentHudongSmMyBinding(constraintLayout, imageView, imageView2, imageView3, imageView4, imageView5, constraintLayout, constraintLayout2, constraintLayout3, findChildViewById, customWebView);
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

    public static FragmentHudongSmMyBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentHudongSmMyBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sm_my, viewGroup, false);
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
