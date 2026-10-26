package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager.widget.ViewPager;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogPagerDrawerBinding implements ViewBinding {
    public final View lineModeLineTop;
    public final ViewPager pager;
    private final LinearLayout rootView;
    public final TabLayout tabLayout;
    public final TextView txtTitle;

    private DialogPagerDrawerBinding(LinearLayout linearLayout, View view, ViewPager viewPager, TabLayout tabLayout, TextView textView) {
        this.rootView = linearLayout;
        this.lineModeLineTop = view;
        this.pager = viewPager;
        this.tabLayout = tabLayout;
        this.txtTitle = textView;
    }

    public static DialogPagerDrawerBinding bind(View view) {
        int i = R.id.line_mode_line_top;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById != null) {
            i = R.id.pager;
            ViewPager viewPager = (ViewPager) ViewBindings.findChildViewById(view, i);
            if (viewPager != null) {
                i = R.id.tabLayout;
                TabLayout tabLayout = (TabLayout) ViewBindings.findChildViewById(view, i);
                if (tabLayout != null) {
                    i = R.id.txtTitle;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        return new DialogPagerDrawerBinding((LinearLayout) view, findChildViewById, viewPager, tabLayout, textView);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogPagerDrawerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogPagerDrawerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_pager_drawer, viewGroup, false);
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
