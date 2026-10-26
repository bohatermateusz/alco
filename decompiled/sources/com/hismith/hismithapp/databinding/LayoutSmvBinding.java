package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class LayoutSmvBinding implements ViewBinding {
    public final FrameLayout layoutFuns2Max;
    public final FrameLayout layoutSpeedMax;
    public final ProgressBar progressFuns2;
    public final ProgressBar progressSpeed;
    private final LinearLayout rootView;
    public final TextView txtFuns2;
    public final TextView txtFuns2Label;
    public final AppCompatTextView txtFuns2Max;
    public final TextView txtSpeed;
    public final TextView txtSpeedLabel;
    public final AppCompatTextView txtSpeedMax;

    private LayoutSmvBinding(LinearLayout linearLayout, FrameLayout frameLayout, FrameLayout frameLayout2, ProgressBar progressBar, ProgressBar progressBar2, TextView textView, TextView textView2, AppCompatTextView appCompatTextView, TextView textView3, TextView textView4, AppCompatTextView appCompatTextView2) {
        this.rootView = linearLayout;
        this.layoutFuns2Max = frameLayout;
        this.layoutSpeedMax = frameLayout2;
        this.progressFuns2 = progressBar;
        this.progressSpeed = progressBar2;
        this.txtFuns2 = textView;
        this.txtFuns2Label = textView2;
        this.txtFuns2Max = appCompatTextView;
        this.txtSpeed = textView3;
        this.txtSpeedLabel = textView4;
        this.txtSpeedMax = appCompatTextView2;
    }

    public static LayoutSmvBinding bind(View view) {
        int i = R.id.layout_funs2_max;
        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
        if (frameLayout != null) {
            i = R.id.layout_speed_max;
            FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout2 != null) {
                i = R.id.progress_funs2;
                ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                if (progressBar != null) {
                    i = R.id.progress_speed;
                    ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                    if (progressBar2 != null) {
                        i = R.id.txt_funs2;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.txt_funs2_label;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.txt_funs2_max;
                                AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                if (appCompatTextView != null) {
                                    i = R.id.txt_speed;
                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView3 != null) {
                                        i = R.id.txt_speed_label;
                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView4 != null) {
                                            i = R.id.txt_speed_max;
                                            AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                            if (appCompatTextView2 != null) {
                                                return new LayoutSmvBinding((LinearLayout) view, frameLayout, frameLayout2, progressBar, progressBar2, textView, textView2, appCompatTextView, textView3, textView4, appCompatTextView2);
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

    public static LayoutSmvBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static LayoutSmvBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.layout_smv, viewGroup, false);
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
