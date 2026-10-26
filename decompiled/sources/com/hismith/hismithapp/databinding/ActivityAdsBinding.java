package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomWebView;

/* loaded from: classes3.dex */
public final class ActivityAdsBinding implements ViewBinding {
    private final FrameLayout rootView;
    public final TextView txtSkip;
    public final CustomWebView webAds;

    private ActivityAdsBinding(FrameLayout frameLayout, TextView textView, CustomWebView customWebView) {
        this.rootView = frameLayout;
        this.txtSkip = textView;
        this.webAds = customWebView;
    }

    public static ActivityAdsBinding bind(View view) {
        int i = R.id.txt_skip;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.web_ads;
            CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
            if (customWebView != null) {
                return new ActivityAdsBinding((FrameLayout) view, textView, customWebView);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityAdsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityAdsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_ads, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public FrameLayout getRoot() {
        return this.rootView;
    }
}
