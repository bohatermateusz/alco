package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager.widget.ViewPager;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityPlayModeBinding implements ViewBinding {
    public final EditText editModeName;
    public final ImageView imgFilter;
    public final ImageView modeImgBluetooth;
    public final ImageView modeImgTabsLeft;
    public final ImageView modeImgTabsRight;
    public final ImageView modeMyImgAdd;
    public final TabLayout modeTabs;
    public final ViewPager modeViewPager;
    private final FrameLayout rootView;
    public final TextView titleTxtName;

    private ActivityPlayModeBinding(FrameLayout frameLayout, EditText editText, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, TabLayout tabLayout, ViewPager viewPager, TextView textView) {
        this.rootView = frameLayout;
        this.editModeName = editText;
        this.imgFilter = imageView;
        this.modeImgBluetooth = imageView2;
        this.modeImgTabsLeft = imageView3;
        this.modeImgTabsRight = imageView4;
        this.modeMyImgAdd = imageView5;
        this.modeTabs = tabLayout;
        this.modeViewPager = viewPager;
        this.titleTxtName = textView;
    }

    public static ActivityPlayModeBinding bind(View view) {
        int i = R.id.edit_mode_name;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.img_filter;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.mode_img_bluetooth;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.mode_img_tabs_left;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.mode_img_tabs_right;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.mode_my_img_add;
                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView5 != null) {
                                i = R.id.mode_tabs;
                                TabLayout tabLayout = (TabLayout) ViewBindings.findChildViewById(view, i);
                                if (tabLayout != null) {
                                    i = R.id.mode_viewPager;
                                    ViewPager viewPager = (ViewPager) ViewBindings.findChildViewById(view, i);
                                    if (viewPager != null) {
                                        i = R.id.title_txt_name;
                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView != null) {
                                            return new ActivityPlayModeBinding((FrameLayout) view, editText, imageView, imageView2, imageView3, imageView4, imageView5, tabLayout, viewPager, textView);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPlayModeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlayModeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_mode, viewGroup, false);
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
