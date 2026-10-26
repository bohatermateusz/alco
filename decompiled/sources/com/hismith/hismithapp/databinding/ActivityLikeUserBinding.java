package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityLikeUserBinding implements ViewBinding {
    public final FrameLayout likeLayout;
    public final RecyclerView likeRecyclerView;
    public final SmartRefreshLayout likeRefreshLayout;
    private final LinearLayout rootView;
    public final LinearLayout titleLayout;
    public final TextView titleTxtName;

    private ActivityLikeUserBinding(LinearLayout linearLayout, FrameLayout frameLayout, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, LinearLayout linearLayout2, TextView textView) {
        this.rootView = linearLayout;
        this.likeLayout = frameLayout;
        this.likeRecyclerView = recyclerView;
        this.likeRefreshLayout = smartRefreshLayout;
        this.titleLayout = linearLayout2;
        this.titleTxtName = textView;
    }

    public static ActivityLikeUserBinding bind(View view) {
        int i = R.id.like_layout;
        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
        if (frameLayout != null) {
            i = R.id.like_recycler_view;
            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
            if (recyclerView != null) {
                i = R.id.like_refreshLayout;
                SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                if (smartRefreshLayout != null) {
                    i = R.id.title_layout;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.title_txt_name;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            return new ActivityLikeUserBinding((LinearLayout) view, frameLayout, recyclerView, smartRefreshLayout, linearLayout, textView);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityLikeUserBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityLikeUserBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_like_user, viewGroup, false);
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
