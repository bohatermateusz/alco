package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.ObservableWebView;

/* loaded from: classes3.dex */
public final class ActivityShopBinding implements ViewBinding {
    public final LinearLayout layoutRoot;
    private final LinearLayout rootView;
    public final ObservableWebView shopWebView;
    public final ProgressBar titleProgress;
    public final View titleView;

    private ActivityShopBinding(LinearLayout linearLayout, LinearLayout linearLayout2, ObservableWebView observableWebView, ProgressBar progressBar, View view) {
        this.rootView = linearLayout;
        this.layoutRoot = linearLayout2;
        this.shopWebView = observableWebView;
        this.titleProgress = progressBar;
        this.titleView = view;
    }

    public static ActivityShopBinding bind(View view) {
        View findChildViewById;
        LinearLayout linearLayout = (LinearLayout) view;
        int i = R.id.shop_webView;
        ObservableWebView observableWebView = (ObservableWebView) ViewBindings.findChildViewById(view, i);
        if (observableWebView != null) {
            i = R.id.title_progress;
            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
            if (progressBar != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.title_view))) != null) {
                return new ActivityShopBinding(linearLayout, linearLayout, observableWebView, progressBar, findChildViewById);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityShopBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityShopBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_shop, viewGroup, false);
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
