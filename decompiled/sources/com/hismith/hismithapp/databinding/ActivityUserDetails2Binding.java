package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.appcompat.widget.Toolbar;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager.widget.ViewPager;
import com.google.android.material.appbar.AppBarLayout;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityUserDetails2Binding implements ViewBinding {
    public final AppBarLayout appbarLayout;
    public final ViewPager detailsFragmentViewPager;
    public final SmartRefreshLayout detailsRefreshLayout;
    public final TabLayout detailsTabs;
    private final FrameLayout rootView;
    public final Toolbar titleBar;

    private ActivityUserDetails2Binding(FrameLayout frameLayout, AppBarLayout appBarLayout, ViewPager viewPager, SmartRefreshLayout smartRefreshLayout, TabLayout tabLayout, Toolbar toolbar) {
        this.rootView = frameLayout;
        this.appbarLayout = appBarLayout;
        this.detailsFragmentViewPager = viewPager;
        this.detailsRefreshLayout = smartRefreshLayout;
        this.detailsTabs = tabLayout;
        this.titleBar = toolbar;
    }

    public static ActivityUserDetails2Binding bind(View view) {
        int i = R.id.appbar_layout;
        AppBarLayout appBarLayout = (AppBarLayout) ViewBindings.findChildViewById(view, i);
        if (appBarLayout != null) {
            i = R.id.details_fragment_view_pager;
            ViewPager viewPager = (ViewPager) ViewBindings.findChildViewById(view, i);
            if (viewPager != null) {
                i = R.id.details_refreshLayout;
                SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                if (smartRefreshLayout != null) {
                    i = R.id.details_tabs;
                    TabLayout tabLayout = (TabLayout) ViewBindings.findChildViewById(view, i);
                    if (tabLayout != null) {
                        i = R.id.title_bar;
                        Toolbar toolbar = (Toolbar) ViewBindings.findChildViewById(view, i);
                        if (toolbar != null) {
                            return new ActivityUserDetails2Binding((FrameLayout) view, appBarLayout, viewPager, smartRefreshLayout, tabLayout, toolbar);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityUserDetails2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityUserDetails2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_user_details2, viewGroup, false);
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
