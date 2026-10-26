package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.MyViewPager;
import com.hismith.hismithapp.view.SlidingTabLayout;

/* loaded from: classes3.dex */
public final class ActivityCommunitySearchBinding implements ViewBinding {
    public final EditText communityEditSearch;
    public final FrameLayout communityLayout;
    public final LinearLayout communityLayoutSearch;
    public final FrameLayout communityLayoutTabs;
    public final SlidingTabLayout communityTabs;
    public final TextView communityTxtCancel;
    public final MyViewPager communityViewPager;
    private final FrameLayout rootView;

    private ActivityCommunitySearchBinding(FrameLayout frameLayout, EditText editText, FrameLayout frameLayout2, LinearLayout linearLayout, FrameLayout frameLayout3, SlidingTabLayout slidingTabLayout, TextView textView, MyViewPager myViewPager) {
        this.rootView = frameLayout;
        this.communityEditSearch = editText;
        this.communityLayout = frameLayout2;
        this.communityLayoutSearch = linearLayout;
        this.communityLayoutTabs = frameLayout3;
        this.communityTabs = slidingTabLayout;
        this.communityTxtCancel = textView;
        this.communityViewPager = myViewPager;
    }

    public static ActivityCommunitySearchBinding bind(View view) {
        int i = R.id.community_edit_search;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            FrameLayout frameLayout = (FrameLayout) view;
            i = R.id.community_layout_search;
            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
            if (linearLayout != null) {
                i = R.id.community_layout_tabs;
                FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout2 != null) {
                    i = R.id.community_tabs;
                    SlidingTabLayout slidingTabLayout = (SlidingTabLayout) ViewBindings.findChildViewById(view, i);
                    if (slidingTabLayout != null) {
                        i = R.id.community_txt_cancel;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.community_view_pager;
                            MyViewPager myViewPager = (MyViewPager) ViewBindings.findChildViewById(view, i);
                            if (myViewPager != null) {
                                return new ActivityCommunitySearchBinding(frameLayout, editText, frameLayout, linearLayout, frameLayout2, slidingTabLayout, textView, myViewPager);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityCommunitySearchBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityCommunitySearchBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_community_search, viewGroup, false);
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
