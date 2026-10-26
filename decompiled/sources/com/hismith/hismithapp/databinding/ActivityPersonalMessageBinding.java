package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.SlidingTabLayout;

/* loaded from: classes3.dex */
public final class ActivityPersonalMessageBinding implements ViewBinding {
    public final SlidingTabLayout messageTabLayout;
    public final ViewPager messageViewPager;
    private final LinearLayout rootView;
    public final LinearLayout titleLayout;
    public final TextView titleTxtName;

    private ActivityPersonalMessageBinding(LinearLayout linearLayout, SlidingTabLayout slidingTabLayout, ViewPager viewPager, LinearLayout linearLayout2, TextView textView) {
        this.rootView = linearLayout;
        this.messageTabLayout = slidingTabLayout;
        this.messageViewPager = viewPager;
        this.titleLayout = linearLayout2;
        this.titleTxtName = textView;
    }

    public static ActivityPersonalMessageBinding bind(View view) {
        int i = R.id.message_tab_layout;
        SlidingTabLayout slidingTabLayout = (SlidingTabLayout) ViewBindings.findChildViewById(view, i);
        if (slidingTabLayout != null) {
            i = R.id.message_view_pager;
            ViewPager viewPager = (ViewPager) ViewBindings.findChildViewById(view, i);
            if (viewPager != null) {
                i = R.id.title_layout;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.title_txt_name;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        return new ActivityPersonalMessageBinding((LinearLayout) view, slidingTabLayout, viewPager, linearLayout, textView);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPersonalMessageBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPersonalMessageBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_personal_message, viewGroup, false);
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
