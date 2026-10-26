package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class FragmentPersonalFansBinding implements ViewBinding {
    public final FrameLayout fansLayout;
    public final SlideRecyclerView fansRecyclerView;
    public final SmartRefreshLayout fansRefreshLayout;
    private final FrameLayout rootView;

    private FragmentPersonalFansBinding(FrameLayout frameLayout, FrameLayout frameLayout2, SlideRecyclerView slideRecyclerView, SmartRefreshLayout smartRefreshLayout) {
        this.rootView = frameLayout;
        this.fansLayout = frameLayout2;
        this.fansRecyclerView = slideRecyclerView;
        this.fansRefreshLayout = smartRefreshLayout;
    }

    public static FragmentPersonalFansBinding bind(View view) {
        FrameLayout frameLayout = (FrameLayout) view;
        int i = R.id.fans_recycler_view;
        SlideRecyclerView slideRecyclerView = (SlideRecyclerView) ViewBindings.findChildViewById(view, i);
        if (slideRecyclerView != null) {
            i = R.id.fans_refreshLayout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                return new FragmentPersonalFansBinding(frameLayout, frameLayout, slideRecyclerView, smartRefreshLayout);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentPersonalFansBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentPersonalFansBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_fans, viewGroup, false);
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
