package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomWebView;

/* loaded from: classes3.dex */
public final class ActivityPrivacyNoticeBinding implements ViewBinding {
    public final TitleBarBinding include;
    public final CustomWebView privacyNoticeWebView;
    private final LinearLayout rootView;
    public final ProgressBar titleProgress;

    private ActivityPrivacyNoticeBinding(LinearLayout linearLayout, TitleBarBinding titleBarBinding, CustomWebView customWebView, ProgressBar progressBar) {
        this.rootView = linearLayout;
        this.include = titleBarBinding;
        this.privacyNoticeWebView = customWebView;
        this.titleProgress = progressBar;
    }

    public static ActivityPrivacyNoticeBinding bind(View view) {
        int i = R.id.include;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById != null) {
            TitleBarBinding bind = TitleBarBinding.bind(findChildViewById);
            int i2 = R.id.privacyNotice_webView;
            CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i2);
            if (customWebView != null) {
                i2 = R.id.title_progress;
                ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i2);
                if (progressBar != null) {
                    return new ActivityPrivacyNoticeBinding((LinearLayout) view, bind, customWebView, progressBar);
                }
            }
            i = i2;
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPrivacyNoticeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPrivacyNoticeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_privacy_notice, viewGroup, false);
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
