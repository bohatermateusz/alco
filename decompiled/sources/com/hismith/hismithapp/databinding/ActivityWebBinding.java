package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomWebView;

/* loaded from: classes3.dex */
public final class ActivityWebBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final ProgressBar titleProgress;
    public final TextView titleTxtName;
    public final CustomWebView webWebView;

    private ActivityWebBinding(LinearLayout linearLayout, ProgressBar progressBar, TextView textView, CustomWebView customWebView) {
        this.rootView = linearLayout;
        this.titleProgress = progressBar;
        this.titleTxtName = textView;
        this.webWebView = customWebView;
    }

    public static ActivityWebBinding bind(View view) {
        int i = R.id.title_progress;
        ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
        if (progressBar != null) {
            i = R.id.title_txt_name;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.web_webView;
                CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                if (customWebView != null) {
                    return new ActivityWebBinding((LinearLayout) view, progressBar, textView, customWebView);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityWebBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityWebBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_web, viewGroup, false);
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
