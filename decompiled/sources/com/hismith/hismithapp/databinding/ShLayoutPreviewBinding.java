package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.image_preview.view.HackyViewPager;

/* loaded from: classes3.dex */
public final class ShLayoutPreviewBinding implements ViewBinding {
    public final Button btnShowOrigin;
    public final FrameLayout fmCenterProgressContainer;
    public final FrameLayout fmImageShowOriginContainer;
    public final ImageView imgCloseButton;
    public final ImageView imgDownload;
    public final RelativeLayout rootView;
    private final RelativeLayout rootView_;
    public final TextView tvId;
    public final TextView tvIndicator;
    public final HackyViewPager viewPager;

    private ShLayoutPreviewBinding(RelativeLayout relativeLayout, Button button, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView, ImageView imageView2, RelativeLayout relativeLayout2, TextView textView, TextView textView2, HackyViewPager hackyViewPager) {
        this.rootView_ = relativeLayout;
        this.btnShowOrigin = button;
        this.fmCenterProgressContainer = frameLayout;
        this.fmImageShowOriginContainer = frameLayout2;
        this.imgCloseButton = imageView;
        this.imgDownload = imageView2;
        this.rootView = relativeLayout2;
        this.tvId = textView;
        this.tvIndicator = textView2;
        this.viewPager = hackyViewPager;
    }

    public static ShLayoutPreviewBinding bind(View view) {
        int i = R.id.btn_show_origin;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.fm_center_progress_container;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.fm_image_show_origin_container;
                FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout2 != null) {
                    i = R.id.imgCloseButton;
                    ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView != null) {
                        i = R.id.img_download;
                        ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView2 != null) {
                            RelativeLayout relativeLayout = (RelativeLayout) view;
                            i = R.id.tv_id;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.tv_indicator;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    i = R.id.viewPager;
                                    HackyViewPager hackyViewPager = (HackyViewPager) ViewBindings.findChildViewById(view, i);
                                    if (hackyViewPager != null) {
                                        return new ShLayoutPreviewBinding(relativeLayout, button, frameLayout, frameLayout2, imageView, imageView2, relativeLayout, textView, textView2, hackyViewPager);
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

    public static ShLayoutPreviewBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ShLayoutPreviewBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.sh_layout_preview, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public RelativeLayout getRoot() {
        return this.rootView_;
    }
}
