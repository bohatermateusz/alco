package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class FragmentPersonalFocusBinding implements ViewBinding {
    public final FrameLayout focusLayout;
    public final RecyclerView focusRecyclerView;
    public final SmartRefreshLayout focusRefreshLayout;
    private final FrameLayout rootView;

    private FragmentPersonalFocusBinding(FrameLayout frameLayout, FrameLayout frameLayout2, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout) {
        this.rootView = frameLayout;
        this.focusLayout = frameLayout2;
        this.focusRecyclerView = recyclerView;
        this.focusRefreshLayout = smartRefreshLayout;
    }

    public static FragmentPersonalFocusBinding bind(View view) {
        FrameLayout frameLayout = (FrameLayout) view;
        int i = R.id.focus_recycler_view;
        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
        if (recyclerView != null) {
            i = R.id.focus_refreshLayout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                return new FragmentPersonalFocusBinding(frameLayout, frameLayout, recyclerView, smartRefreshLayout);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentPersonalFocusBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentPersonalFocusBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_focus, viewGroup, false);
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
