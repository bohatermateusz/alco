package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityPlayStatisticsBinding implements ViewBinding {
    public final TitleBarBinding layoutTitle;
    public final RecyclerView recyclerData;
    public final SmartRefreshLayout refreshLayout;
    private final ConstraintLayout rootView;

    private ActivityPlayStatisticsBinding(ConstraintLayout constraintLayout, TitleBarBinding titleBarBinding, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout) {
        this.rootView = constraintLayout;
        this.layoutTitle = titleBarBinding;
        this.recyclerData = recyclerView;
        this.refreshLayout = smartRefreshLayout;
    }

    public static ActivityPlayStatisticsBinding bind(View view) {
        int i = R.id.layout_title;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById != null) {
            TitleBarBinding bind = TitleBarBinding.bind(findChildViewById);
            int i2 = R.id.recycler_data;
            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i2);
            if (recyclerView != null) {
                i2 = R.id.refresh_layout;
                SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i2);
                if (smartRefreshLayout != null) {
                    return new ActivityPlayStatisticsBinding((ConstraintLayout) view, bind, recyclerView, smartRefreshLayout);
                }
            }
            i = i2;
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPlayStatisticsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlayStatisticsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_statistics, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public ConstraintLayout getRoot() {
        return this.rootView;
    }
}
