package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager.widget.ViewPager;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityPersonalActionBinding implements ViewBinding {
    public final TabLayout personalTabLayout;
    public final ViewPager personalViewpager;
    private final LinearLayout rootView;

    private ActivityPersonalActionBinding(LinearLayout linearLayout, TabLayout tabLayout, ViewPager viewPager) {
        this.rootView = linearLayout;
        this.personalTabLayout = tabLayout;
        this.personalViewpager = viewPager;
    }

    public static ActivityPersonalActionBinding bind(View view) {
        int i = R.id.personal_tab_layout;
        TabLayout tabLayout = (TabLayout) ViewBindings.findChildViewById(view, i);
        if (tabLayout != null) {
            i = R.id.personal_viewpager;
            ViewPager viewPager = (ViewPager) ViewBindings.findChildViewById(view, i);
            if (viewPager != null) {
                return new ActivityPersonalActionBinding((LinearLayout) view, tabLayout, viewPager);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPersonalActionBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPersonalActionBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_personal_action, viewGroup, false);
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
