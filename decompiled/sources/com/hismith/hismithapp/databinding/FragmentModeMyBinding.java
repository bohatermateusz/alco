package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class FragmentModeMyBinding implements ViewBinding {
    public final ImageView modeMyImgEmpty;
    public final RecyclerView modeMyRecyclerView;
    public final SmartRefreshLayout modeMyRefreshLayout;
    private final ConstraintLayout rootView;

    private FragmentModeMyBinding(ConstraintLayout constraintLayout, ImageView imageView, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout) {
        this.rootView = constraintLayout;
        this.modeMyImgEmpty = imageView;
        this.modeMyRecyclerView = recyclerView;
        this.modeMyRefreshLayout = smartRefreshLayout;
    }

    public static FragmentModeMyBinding bind(View view) {
        int i = R.id.mode_my_img_empty;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.mode_my_recycler_view;
            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
            if (recyclerView != null) {
                i = R.id.mode_my_refreshLayout;
                SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                if (smartRefreshLayout != null) {
                    return new FragmentModeMyBinding((ConstraintLayout) view, imageView, recyclerView, smartRefreshLayout);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentModeMyBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentModeMyBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_mode_my, viewGroup, false);
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
