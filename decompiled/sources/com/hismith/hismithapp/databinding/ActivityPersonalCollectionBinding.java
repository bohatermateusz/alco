package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.SlidingTabLayout;

/* loaded from: classes3.dex */
public final class ActivityPersonalCollectionBinding implements ViewBinding {
    public final SlidingTabLayout collectionTabLayout;
    public final ViewPager collectionViewPager;
    private final LinearLayout rootView;

    private ActivityPersonalCollectionBinding(LinearLayout linearLayout, SlidingTabLayout slidingTabLayout, ViewPager viewPager) {
        this.rootView = linearLayout;
        this.collectionTabLayout = slidingTabLayout;
        this.collectionViewPager = viewPager;
    }

    public static ActivityPersonalCollectionBinding bind(View view) {
        int i = R.id.collection_tab_layout;
        SlidingTabLayout slidingTabLayout = (SlidingTabLayout) ViewBindings.findChildViewById(view, i);
        if (slidingTabLayout != null) {
            i = R.id.collection_view_pager;
            ViewPager viewPager = (ViewPager) ViewBindings.findChildViewById(view, i);
            if (viewPager != null) {
                return new ActivityPersonalCollectionBinding((LinearLayout) view, slidingTabLayout, viewPager);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPersonalCollectionBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPersonalCollectionBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_personal_collection, viewGroup, false);
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
