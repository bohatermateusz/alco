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
public final class FragmentCommunityTopicBinding implements ViewBinding {
    public final FrameLayout communityLayout;
    public final RecyclerView communityRecycler;
    public final SmartRefreshLayout communityRefreshLayout;
    private final FrameLayout rootView;

    private FragmentCommunityTopicBinding(FrameLayout frameLayout, FrameLayout frameLayout2, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout) {
        this.rootView = frameLayout;
        this.communityLayout = frameLayout2;
        this.communityRecycler = recyclerView;
        this.communityRefreshLayout = smartRefreshLayout;
    }

    public static FragmentCommunityTopicBinding bind(View view) {
        FrameLayout frameLayout = (FrameLayout) view;
        int i = R.id.community_recycler;
        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
        if (recyclerView != null) {
            i = R.id.community_refreshLayout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                return new FragmentCommunityTopicBinding(frameLayout, frameLayout, recyclerView, smartRefreshLayout);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentCommunityTopicBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentCommunityTopicBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_community_topic, viewGroup, false);
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
