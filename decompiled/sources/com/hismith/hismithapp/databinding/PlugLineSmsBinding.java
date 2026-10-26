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
public final class PlugLineSmsBinding implements ViewBinding {
    public final ImageView imgRunMode;
    public final LinearLayout layoutPositionLabel;
    public final FrameLayout layoutPositionMax;
    public final FrameLayout layoutPositionMin;
    public final RelativeLayout layoutPositionProgress;
    public final LinearLayout layoutSmoothLabel;
    public final LinearLayout layoutSpeedLabel;
    public final FrameLayout layoutSpeedMax;
    public final ConstraintLayout layoutSpeedProgress;
    public final ProgressBar progressPosition;
    public final ProgressBar progressSpeed;
    private final ConstraintLayout rootView;
    public final CustomSeekbar seekSmooth;
    public final TextView txtPosition;
    public final TextView txtPositionLabel;
    public final TextView txtPositionMax;
    public final TextView txtPositionMin;
    public final TextView txtSmooth;
    public final TextView txtSmoothLabel;
    public final TextView txtSpeed;
    public final TextView txtSpeedLabel;
    public final AppCompatTextView txtSpeedMax;

    private PlugLineSmsBinding(ConstraintLayout constraintLayout, ImageView imageView, LinearLayout linearLayout, FrameLayout frameLayout, FrameLayout frameLayout2, RelativeLayout relativeLayout, LinearLayout linearLayout2, LinearLayout linearLayout3, FrameLayout frameLayout3, ConstraintLayout constraintLayout2, ProgressBar progressBar, ProgressBar progressBar2, CustomSeekbar customSeekbar, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, AppCompatTextView appCompatTextView) {
        this.rootView = constraintLayout;
        this.imgRunMode = imageView;
        this.layoutPositionLabel = linearLayout;
        this.layoutPositionMax = frameLayout;
        this.layoutPositionMin = frameLayout2;
        this.layoutPositionProgress = relativeLayout;
        this.layoutSmoothLabel = linearLayout2;
        this.layoutSpeedLabel = linearLayout3;
        this.layoutSpeedMax = frameLayout3;
        this.layoutSpeedProgress = constraintLayout2;
        this.progressPosition = progressBar;
        this.progressSpeed = progressBar2;
        this.seekSmooth = customSeekbar;
        this.txtPosition = textView;
        this.txtPositionLabel = textView2;
        this.txtPositionMax = textView3;
        this.txtPositionMin = textView4;
        this.txtSmooth = textView5;
        this.txtSmoothLabel = textView6;
        this.txtSpeed = textView7;
        this.txtSpeedLabel = textView8;
        this.txtSpeedMax = appCompatTextView;
    }

    public static PlugLineSmsBinding bind(View view) {
        int i = R.id.img_run_mode;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.layout_position_label;
            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
            if (linearLayout != null) {
                i = R.id.layout_position_max;
                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout != null) {
                    i = R.id.layout_position_min;
                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                    if (frameLayout2 != null) {
                        i = R.id.layout_position_progress;
                        RelativeLayout relativeLayout = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                        if (relativeLayout != null) {
                            i = R.id.layout_smooth_label;
                            LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout2 != null) {
                                i = R.id.layout_speed_label;
                                LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout3 != null) {
                                    i = R.id.layout_speed_max;
                                    FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                    if (frameLayout3 != null) {
                                        i = R.id.layout_speed_progress;
                                        ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                        if (constraintLayout != null) {
                                            i = R.id.progress_position;
                                            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                            if (progressBar != null) {
                                                i = R.id.progress_speed;
                                                ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                if (progressBar2 != null) {
                                                    i = R.id.seek_smooth;
                                                    CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
                                                    if (customSeekbar != null) {
                                                        i = R.id.txt_position;
                                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView != null) {
                                                            i = R.id.txt_position_label;
                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView2 != null) {
                                                                i = R.id.txt_position_max;
                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView3 != null) {
                                                                    i = R.id.txt_position_min;
                                                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView4 != null) {
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
                                                                                        AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (appCompatTextView != null) {
                                                                                            return new PlugLineSmsBinding((ConstraintLayout) view, imageView, linearLayout, frameLayout, frameLayout2, relativeLayout, linearLayout2, linearLayout3, frameLayout3, constraintLayout, progressBar, progressBar2, customSeekbar, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, appCompatTextView);
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

    public static PlugLineSmsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PlugLineSmsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.plug_line_sms, viewGroup, false);
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
