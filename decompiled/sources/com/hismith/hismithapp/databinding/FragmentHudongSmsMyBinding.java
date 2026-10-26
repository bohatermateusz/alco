package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomWebView;

/* loaded from: classes3.dex */
public final class FragmentHudongSmsMyBinding implements ViewBinding {
    public final ImageView imgAuto;
    public final ImageView imgSpeedMode;
    public final ImageView imgSpeedModeBg;
    public final ConstraintLayout layoutPositionBox;
    public final ConstraintLayout layoutPositionMax;
    public final ConstraintLayout layoutPositionMin;
    public final ConstraintLayout layoutRoot;
    public final ConstraintLayout layoutSpeedMode;
    public final View line1;
    public final Guideline lineMaxStart;
    public final Guideline lineMinStart;
    public final ProgressBar progressPosition;
    private final ConstraintLayout rootView;
    public final AppCompatTextView txtPositionMax;
    public final AppCompatTextView txtPositionMin;
    public final CustomWebView webDevice;

    private FragmentHudongSmsMyBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, ConstraintLayout constraintLayout4, ConstraintLayout constraintLayout5, ConstraintLayout constraintLayout6, View view, Guideline guideline, Guideline guideline2, ProgressBar progressBar, AppCompatTextView appCompatTextView, AppCompatTextView appCompatTextView2, CustomWebView customWebView) {
        this.rootView = constraintLayout;
        this.imgAuto = imageView;
        this.imgSpeedMode = imageView2;
        this.imgSpeedModeBg = imageView3;
        this.layoutPositionBox = constraintLayout2;
        this.layoutPositionMax = constraintLayout3;
        this.layoutPositionMin = constraintLayout4;
        this.layoutRoot = constraintLayout5;
        this.layoutSpeedMode = constraintLayout6;
        this.line1 = view;
        this.lineMaxStart = guideline;
        this.lineMinStart = guideline2;
        this.progressPosition = progressBar;
        this.txtPositionMax = appCompatTextView;
        this.txtPositionMin = appCompatTextView2;
        this.webDevice = customWebView;
    }

    public static FragmentHudongSmsMyBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_auto;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_speed_mode;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_speed_mode_bg;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.layout_position_box;
                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                    if (constraintLayout != null) {
                        i = R.id.layout_position_max;
                        ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                        if (constraintLayout2 != null) {
                            i = R.id.layout_position_min;
                            ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout3 != null) {
                                ConstraintLayout constraintLayout4 = (ConstraintLayout) view;
                                i = R.id.layout_speed_mode;
                                ConstraintLayout constraintLayout5 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                if (constraintLayout5 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line1))) != null) {
                                    i = R.id.line_max_start;
                                    Guideline guideline = (Guideline) ViewBindings.findChildViewById(view, i);
                                    if (guideline != null) {
                                        i = R.id.line_min_start;
                                        Guideline guideline2 = (Guideline) ViewBindings.findChildViewById(view, i);
                                        if (guideline2 != null) {
                                            i = R.id.progress_position;
                                            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                            if (progressBar != null) {
                                                i = R.id.txt_position_max;
                                                AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                if (appCompatTextView != null) {
                                                    i = R.id.txt_position_min;
                                                    AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                    if (appCompatTextView2 != null) {
                                                        i = R.id.web_device;
                                                        CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                                        if (customWebView != null) {
                                                            return new FragmentHudongSmsMyBinding(constraintLayout4, imageView, imageView2, imageView3, constraintLayout, constraintLayout2, constraintLayout3, constraintLayout4, constraintLayout5, findChildViewById, guideline, guideline2, progressBar, appCompatTextView, appCompatTextView2, customWebView);
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

    public static FragmentHudongSmsMyBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentHudongSmsMyBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sms_my, viewGroup, false);
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
