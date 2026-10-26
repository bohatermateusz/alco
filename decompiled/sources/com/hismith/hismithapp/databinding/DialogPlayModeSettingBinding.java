package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.SettingProgressView;

/* loaded from: classes3.dex */
public final class DialogPlayModeSettingBinding implements ViewBinding {
    public final LinearLayout dialogConfirmLayoutContent;
    public final ImageView imgBack;
    public final ConstraintLayout layoutPosition;
    public final ConstraintLayout layoutPositionMax;
    public final ConstraintLayout layoutPositionMin;
    public final LinearLayout layoutSlideSpeed;
    public final ConstraintLayout layoutSmooth;
    public final Guideline lineMaxStart;
    public final Guideline lineMinStart;
    public final ProgressBar progressPosition;
    public final SettingProgressView progressSlideSpeed;
    private final LinearLayout rootView;
    public final CustomSeekbar seekSmooth;
    public final AppCompatTextView txtPositionMax;
    public final AppCompatTextView txtPositionMin;
    public final TextView txtSlideSpeed;

    private DialogPlayModeSettingBinding(LinearLayout linearLayout, LinearLayout linearLayout2, ImageView imageView, ConstraintLayout constraintLayout, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, LinearLayout linearLayout3, ConstraintLayout constraintLayout4, Guideline guideline, Guideline guideline2, ProgressBar progressBar, SettingProgressView settingProgressView, CustomSeekbar customSeekbar, AppCompatTextView appCompatTextView, AppCompatTextView appCompatTextView2, TextView textView) {
        this.rootView = linearLayout;
        this.dialogConfirmLayoutContent = linearLayout2;
        this.imgBack = imageView;
        this.layoutPosition = constraintLayout;
        this.layoutPositionMax = constraintLayout2;
        this.layoutPositionMin = constraintLayout3;
        this.layoutSlideSpeed = linearLayout3;
        this.layoutSmooth = constraintLayout4;
        this.lineMaxStart = guideline;
        this.lineMinStart = guideline2;
        this.progressPosition = progressBar;
        this.progressSlideSpeed = settingProgressView;
        this.seekSmooth = customSeekbar;
        this.txtPositionMax = appCompatTextView;
        this.txtPositionMin = appCompatTextView2;
        this.txtSlideSpeed = textView;
    }

    public static DialogPlayModeSettingBinding bind(View view) {
        int i = R.id.dialog_confirm_layout_content;
        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
        if (linearLayout != null) {
            i = R.id.img_back;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.layout_position;
                ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                if (constraintLayout != null) {
                    i = R.id.layout_position_max;
                    ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                    if (constraintLayout2 != null) {
                        i = R.id.layout_position_min;
                        ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                        if (constraintLayout3 != null) {
                            i = R.id.layout_slide_speed;
                            LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout2 != null) {
                                i = R.id.layout_smooth;
                                ConstraintLayout constraintLayout4 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                if (constraintLayout4 != null) {
                                    i = R.id.line_max_start;
                                    Guideline guideline = (Guideline) ViewBindings.findChildViewById(view, i);
                                    if (guideline != null) {
                                        i = R.id.line_min_start;
                                        Guideline guideline2 = (Guideline) ViewBindings.findChildViewById(view, i);
                                        if (guideline2 != null) {
                                            i = R.id.progress_position;
                                            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                            if (progressBar != null) {
                                                i = R.id.progress_slide_speed;
                                                SettingProgressView settingProgressView = (SettingProgressView) ViewBindings.findChildViewById(view, i);
                                                if (settingProgressView != null) {
                                                    i = R.id.seek_smooth;
                                                    CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
                                                    if (customSeekbar != null) {
                                                        i = R.id.txt_position_max;
                                                        AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                        if (appCompatTextView != null) {
                                                            i = R.id.txt_position_min;
                                                            AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                            if (appCompatTextView2 != null) {
                                                                i = R.id.txt_slide_speed;
                                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView != null) {
                                                                    return new DialogPlayModeSettingBinding((LinearLayout) view, linearLayout, imageView, constraintLayout, constraintLayout2, constraintLayout3, linearLayout2, constraintLayout4, guideline, guideline2, progressBar, settingProgressView, customSeekbar, appCompatTextView, appCompatTextView2, textView);
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

    public static DialogPlayModeSettingBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogPlayModeSettingBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_play_mode_setting, viewGroup, false);
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
