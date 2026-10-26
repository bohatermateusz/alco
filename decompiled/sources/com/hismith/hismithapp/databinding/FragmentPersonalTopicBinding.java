package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class FragmentPersonalTopicBinding implements ViewBinding {
    public final RecyclerView personalTopicRecyclerView;
    public final SmartRefreshLayout personalTopicRefreshLayout;
    private final LinearLayout rootView;

    private FragmentPersonalTopicBinding(LinearLayout linearLayout, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout) {
        this.rootView = linearLayout;
        this.personalTopicRecyclerView = recyclerView;
        this.personalTopicRefreshLayout = smartRefreshLayout;
    }

    public static FragmentPersonalTopicBinding bind(View view) {
        int i = R.id.personal_topic_recycler_view;
        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
        if (recyclerView != null) {
            i = R.id.personal_topic_refreshLayout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                return new FragmentPersonalTopicBinding((LinearLayout) view, recyclerView, smartRefreshLayout);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentPersonalTopicBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentPersonalTopicBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_topic, viewGroup, false);
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
