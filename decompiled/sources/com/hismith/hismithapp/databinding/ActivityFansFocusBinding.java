package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.NoScrollViewPager;
import com.hismith.hismithapp.view.SlidingTabLayout;

/* loaded from: classes3.dex */
public final class ActivityFansFocusBinding implements ViewBinding {
    public final SlidingTabLayout fansFocusTabs;
    public final NoScrollViewPager fansFocusViewpager;
    private final LinearLayout rootView;
    public final ImageView titleImgAdd;
    public final TextView titleTxtName;

    private ActivityFansFocusBinding(LinearLayout linearLayout, SlidingTabLayout slidingTabLayout, NoScrollViewPager noScrollViewPager, ImageView imageView, TextView textView) {
        this.rootView = linearLayout;
        this.fansFocusTabs = slidingTabLayout;
        this.fansFocusViewpager = noScrollViewPager;
        this.titleImgAdd = imageView;
        this.titleTxtName = textView;
    }

    public static ActivityFansFocusBinding bind(View view) {
        int i = R.id.fans_focus_tabs;
        SlidingTabLayout slidingTabLayout = (SlidingTabLayout) ViewBindings.findChildViewById(view, i);
        if (slidingTabLayout != null) {
            i = R.id.fans_focus_viewpager;
            NoScrollViewPager noScrollViewPager = (NoScrollViewPager) ViewBindings.findChildViewById(view, i);
            if (noScrollViewPager != null) {
                i = R.id.title_img_add;
                ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView != null) {
                    i = R.id.title_txt_name;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        return new ActivityFansFocusBinding((LinearLayout) view, slidingTabLayout, noScrollViewPager, imageView, textView);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityFansFocusBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityFansFocusBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_fans_focus, viewGroup, false);
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
