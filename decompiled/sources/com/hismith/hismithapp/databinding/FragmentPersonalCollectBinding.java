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
public final class FragmentPersonalCollectBinding implements ViewBinding {
    public final RecyclerView personalCollectRecyclerView;
    public final SmartRefreshLayout personalCollectRefreshLayout;
    private final LinearLayout rootView;

    private FragmentPersonalCollectBinding(LinearLayout linearLayout, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout) {
        this.rootView = linearLayout;
        this.personalCollectRecyclerView = recyclerView;
        this.personalCollectRefreshLayout = smartRefreshLayout;
    }

    public static FragmentPersonalCollectBinding bind(View view) {
        int i = R.id.personal_collect_recycler_view;
        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
        if (recyclerView != null) {
            i = R.id.personal_collect_refreshLayout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                return new FragmentPersonalCollectBinding((LinearLayout) view, recyclerView, smartRefreshLayout);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentPersonalCollectBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentPersonalCollectBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_collect, viewGroup, false);
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
