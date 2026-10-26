package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Group;
import androidx.constraintlayout.widget.Guideline;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomSeekbar;

/* loaded from: classes3.dex */
public final class LayoutDevicePanelBinding implements ViewBinding {
    public final Group groupSms;
    public final Group groupSmv;
    public final View guideView;
    public final LinearLayout layoutFuns1Label;
    public final ConstraintLayout layoutFuns1Value;
    public final LinearLayout layoutFuns2Label;
    public final FrameLayout layoutFuns2Max;
    public final ConstraintLayout layoutFuns2Value;
    public final LinearLayout layoutPositionLabel;
    public final ConstraintLayout layoutPositionMax;
    public final ConstraintLayout layoutPositionMin;
    public final ConstraintLayout layoutPositionValue;
    public final LinearLayout layoutSmoothLabel;
    public final FrameLayout layoutSpeedMax;
    public final Guideline lineMaxStart;
    public final Guideline lineMinStart;
    public final ProgressBar progressFuns2;
    public final ProgressBar progressPosition;
    public final ProgressBar progressSpeed;
    private final ConstraintLayout rootView;
    public final CustomSeekbar seekSmooth;
    public final TextView txtFuns2;
    public final TextView txtFuns2Label;
    public final AppCompatTextView txtFuns2Max;
    public final TextView txtPosition;
    public final TextView txtPositionLabel;
    public final AppCompatTextView txtPositionMax;
    public final AppCompatTextView txtPositionMin;
    public final TextView txtSmooth;
    public final TextView txtSmoothLabel;
    public final TextView txtSpeed;
    public final TextView txtSpeedLabel;
    public final AppCompatTextView txtSpeedMax;

    private LayoutDevicePanelBinding(ConstraintLayout constraintLayout, Group group, Group group2, View view, LinearLayout linearLayout, ConstraintLayout constraintLayout2, LinearLayout linearLayout2, FrameLayout frameLayout, ConstraintLayout constraintLayout3, LinearLayout linearLayout3, ConstraintLayout constraintLayout4, ConstraintLayout constraintLayout5, ConstraintLayout constraintLayout6, LinearLayout linearLayout4, FrameLayout frameLayout2, Guideline guideline, Guideline guideline2, ProgressBar progressBar, ProgressBar progressBar2, ProgressBar progressBar3, CustomSeekbar customSeekbar, TextView textView, TextView textView2, AppCompatTextView appCompatTextView, TextView textView3, TextView textView4, AppCompatTextView appCompatTextView2, AppCompatTextView appCompatTextView3, TextView textView5, TextView textView6, TextView textView7, TextView textView8, AppCompatTextView appCompatTextView4) {
        this.rootView = constraintLayout;
        this.groupSms = group;
        this.groupSmv = group2;
        this.guideView = view;
        this.layoutFuns1Label = linearLayout;
        this.layoutFuns1Value = constraintLayout2;
        this.layoutFuns2Label = linearLayout2;
        this.layoutFuns2Max = frameLayout;
        this.layoutFuns2Value = constraintLayout3;
        this.layoutPositionLabel = linearLayout3;
        this.layoutPositionMax = constraintLayout4;
        this.layoutPositionMin = constraintLayout5;
        this.layoutPositionValue = constraintLayout6;
        this.layoutSmoothLabel = linearLayout4;
        this.layoutSpeedMax = frameLayout2;
        this.lineMaxStart = guideline;
        this.lineMinStart = guideline2;
        this.progressFuns2 = progressBar;
        this.progressPosition = progressBar2;
        this.progressSpeed = progressBar3;
        this.seekSmooth = customSeekbar;
        this.txtFuns2 = textView;
        this.txtFuns2Label = textView2;
        this.txtFuns2Max = appCompatTextView;
        this.txtPosition = textView3;
        this.txtPositionLabel = textView4;
        this.txtPositionMax = appCompatTextView2;
        this.txtPositionMin = appCompatTextView3;
        this.txtSmooth = textView5;
        this.txtSmoothLabel = textView6;
        this.txtSpeed = textView7;
        this.txtSpeedLabel = textView8;
        this.txtSpeedMax = appCompatTextView4;
    }

    public static LayoutDevicePanelBinding bind(View view) {
        View findChildViewById;
        int i = R.id.group_sms;
        Group group = (Group) ViewBindings.findChildViewById(view, i);
        if (group != null) {
            i = R.id.group_smv;
            Group group2 = (Group) ViewBindings.findChildViewById(view, i);
            if (group2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.guide_view))) != null) {
                i = R.id.layout_funs1_label;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.layout_funs1_value;
                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                    if (constraintLayout != null) {
                        i = R.id.layout_funs2_label;
                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout2 != null) {
                            i = R.id.layout_funs2_max;
                            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                            if (frameLayout != null) {
                                i = R.id.layout_funs2_value;
                                ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                if (constraintLayout2 != null) {
                                    i = R.id.layout_position_label;
                                    LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout3 != null) {
                                        i = R.id.layout_position_max;
                                        ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                        if (constraintLayout3 != null) {
                                            i = R.id.layout_position_min;
                                            ConstraintLayout constraintLayout4 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                            if (constraintLayout4 != null) {
                                                i = R.id.layout_position_value;
                                                ConstraintLayout constraintLayout5 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                if (constraintLayout5 != null) {
                                                    i = R.id.layout_smooth_label;
                                                    LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                    if (linearLayout4 != null) {
                                                        i = R.id.layout_speed_max;
                                                        FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                        if (frameLayout2 != null) {
                                                            i = R.id.line_max_start;
                                                            Guideline guideline = (Guideline) ViewBindings.findChildViewById(view, i);
                                                            if (guideline != null) {
                                                                i = R.id.line_min_start;
                                                                Guideline guideline2 = (Guideline) ViewBindings.findChildViewById(view, i);
                                                                if (guideline2 != null) {
                                                                    i = R.id.progress_funs2;
                                                                    ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                    if (progressBar != null) {
                                                                        i = R.id.progress_position;
                                                                        ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                        if (progressBar2 != null) {
                                                                            i = R.id.progress_speed;
                                                                            ProgressBar progressBar3 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                            if (progressBar3 != null) {
                                                                                i = R.id.seek_smooth;
                                                                                CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
                                                                                if (customSeekbar != null) {
                                                                                    i = R.id.txt_funs2;
                                                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView != null) {
                                                                                        i = R.id.txt_funs2_label;
                                                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView2 != null) {
                                                                                            i = R.id.txt_funs2_max;
                                                                                            AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (appCompatTextView != null) {
                                                                                                i = R.id.txt_position;
                                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView3 != null) {
                                                                                                    i = R.id.txt_position_label;
                                                                                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView4 != null) {
                                                                                                        i = R.id.txt_position_max;
                                                                                                        AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (appCompatTextView2 != null) {
                                                                                                            i = R.id.txt_position_min;
                                                                                                            AppCompatTextView appCompatTextView3 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (appCompatTextView3 != null) {
                                                                                                                i = R.id.txt_smooth;
                                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView5 != null) {
                                                                                                                    i = R.id.txt_smooth_label;
                                                                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (textView6 != null) {
                                                                                                                        i = R.id.txt_speed;
                                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (textView7 != null) {
                                                                                                                            i = R.id.txt_speed_label;
                                                                                                                            TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (textView8 != null) {
                                                                                                                                i = R.id.txt_speed_max;
                                                                                                                                AppCompatTextView appCompatTextView4 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (appCompatTextView4 != null) {
                                                                                                                                    return new LayoutDevicePanelBinding((ConstraintLayout) view, group, group2, findChildViewById, linearLayout, constraintLayout, linearLayout2, frameLayout, constraintLayout2, linearLayout3, constraintLayout3, constraintLayout4, constraintLayout5, linearLayout4, frameLayout2, guideline, guideline2, progressBar, progressBar2, progressBar3, customSeekbar, textView, textView2, appCompatTextView, textView3, textView4, appCompatTextView2, appCompatTextView3, textView5, textView6, textView7, textView8, appCompatTextView4);
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
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static LayoutDevicePanelBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static LayoutDevicePanelBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.layout_device_panel, viewGroup, false);
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
