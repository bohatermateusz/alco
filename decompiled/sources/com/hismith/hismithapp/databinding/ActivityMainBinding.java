package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager2.widget.ViewPager2;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityMainBinding implements ViewBinding {
    public final LinearLayout layoutRoot;
    public final ImageView mainImgCommunity;
    public final ImageView mainImgMy;
    public final ImageView mainImgMyDot;
    public final ImageView mainImgPlay;
    public final ImageView mainImgStore;
    public final LinearLayout mainLayoutCommunity;
    public final LinearLayout mainLayoutMenu;
    public final LinearLayout mainLayoutMy;
    public final LinearLayout mainLayoutPlay;
    public final LinearLayout mainLayoutStore;
    public final TextView mainTxtCommunity;
    public final TextView mainTxtMy;
    public final TextView mainTxtPlay;
    public final TextView mainTxtStore;
    public final ViewPager2 mainViewpager;
    private final LinearLayout rootView;

    private ActivityMainBinding(LinearLayout linearLayout, LinearLayout linearLayout2, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, LinearLayout linearLayout3, LinearLayout linearLayout4, LinearLayout linearLayout5, LinearLayout linearLayout6, LinearLayout linearLayout7, TextView textView, TextView textView2, TextView textView3, TextView textView4, ViewPager2 viewPager2) {
        this.rootView = linearLayout;
        this.layoutRoot = linearLayout2;
        this.mainImgCommunity = imageView;
        this.mainImgMy = imageView2;
        this.mainImgMyDot = imageView3;
        this.mainImgPlay = imageView4;
        this.mainImgStore = imageView5;
        this.mainLayoutCommunity = linearLayout3;
        this.mainLayoutMenu = linearLayout4;
        this.mainLayoutMy = linearLayout5;
        this.mainLayoutPlay = linearLayout6;
        this.mainLayoutStore = linearLayout7;
        this.mainTxtCommunity = textView;
        this.mainTxtMy = textView2;
        this.mainTxtPlay = textView3;
        this.mainTxtStore = textView4;
        this.mainViewpager = viewPager2;
    }

    public static ActivityMainBinding bind(View view) {
        LinearLayout linearLayout = (LinearLayout) view;
        int i = R.id.main_img_community;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.main_img_my;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.main_img_my_dot;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.main_img_play;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.main_img_store;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.main_layout_community;
                            LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout2 != null) {
                                i = R.id.main_layout_menu;
                                LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout3 != null) {
                                    i = R.id.main_layout_my;
                                    LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout4 != null) {
                                        i = R.id.main_layout_play;
                                        LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout5 != null) {
                                            i = R.id.main_layout_store;
                                            LinearLayout linearLayout6 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout6 != null) {
                                                i = R.id.main_txt_community;
                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView != null) {
                                                    i = R.id.main_txt_my;
                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView2 != null) {
                                                        i = R.id.main_txt_play;
                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView3 != null) {
                                                            i = R.id.main_txt_store;
                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView4 != null) {
                                                                i = R.id.main_viewpager;
                                                                ViewPager2 viewPager2 = (ViewPager2) ViewBindings.findChildViewById(view, i);
                                                                if (viewPager2 != null) {
                                                                    return new ActivityMainBinding(linearLayout, linearLayout, imageView, imageView2, imageView3, imageView4, imageView5, linearLayout2, linearLayout3, linearLayout4, linearLayout5, linearLayout6, textView, textView2, textView3, textView4, viewPager2);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
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

    public static ActivityMainBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityMainBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_main, viewGroup, false);
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
