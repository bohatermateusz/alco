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
public final class ActivityLoginTermsBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final ProgressBar titleProgress;
    public final TextView titleTxtName;
    public final TextView userTermsTxtAgree;
    public final TextView userTermsTxtDisagree;
    public final CustomWebView userTermsWebView;

    private ActivityLoginTermsBinding(LinearLayout linearLayout, ProgressBar progressBar, TextView textView, TextView textView2, TextView textView3, CustomWebView customWebView) {
        this.rootView = linearLayout;
        this.titleProgress = progressBar;
        this.titleTxtName = textView;
        this.userTermsTxtAgree = textView2;
        this.userTermsTxtDisagree = textView3;
        this.userTermsWebView = customWebView;
    }

    public static ActivityLoginTermsBinding bind(View view) {
        int i = R.id.title_progress;
        ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
        if (progressBar != null) {
            i = R.id.title_txt_name;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.userTerms_txt_agree;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.userTerms_txt_disagree;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        i = R.id.userTerms_webView;
                        CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                        if (customWebView != null) {
                            return new ActivityLoginTermsBinding((LinearLayout) view, progressBar, textView, textView2, textView3, customWebView);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityLoginTermsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityLoginTermsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_login_terms, viewGroup, false);
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
