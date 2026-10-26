package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.MyViewPager;
import com.hismith.hismithapp.view.SlidingTabLayout;

/* loaded from: classes3.dex */
public final class FragmentCommunity2Binding implements ViewBinding {
    public final TextView communityEditSearch;
    public final ImageView communityImgPublish;
    public final FrameLayout communityLayout;
    public final LinearLayout communityLayoutSearch;
    public final SlidingTabLayout communityTabs;
    public final MyViewPager communityViewPager;
    private final FrameLayout rootView;

    private FragmentCommunity2Binding(FrameLayout frameLayout, TextView textView, ImageView imageView, FrameLayout frameLayout2, LinearLayout linearLayout, SlidingTabLayout slidingTabLayout, MyViewPager myViewPager) {
        this.rootView = frameLayout;
        this.communityEditSearch = textView;
        this.communityImgPublish = imageView;
        this.communityLayout = frameLayout2;
        this.communityLayoutSearch = linearLayout;
        this.communityTabs = slidingTabLayout;
        this.communityViewPager = myViewPager;
    }

    public static FragmentCommunity2Binding bind(View view) {
        int i = R.id.community_edit_search;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.community_img_publish;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                FrameLayout frameLayout = (FrameLayout) view;
                i = R.id.community_layout_search;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.community_tabs;
                    SlidingTabLayout slidingTabLayout = (SlidingTabLayout) ViewBindings.findChildViewById(view, i);
                    if (slidingTabLayout != null) {
                        i = R.id.community_view_pager;
                        MyViewPager myViewPager = (MyViewPager) ViewBindings.findChildViewById(view, i);
                        if (myViewPager != null) {
                            return new FragmentCommunity2Binding(frameLayout, textView, imageView, frameLayout, linearLayout, slidingTabLayout, myViewPager);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentCommunity2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentCommunity2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_community2, viewGroup, false);
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
