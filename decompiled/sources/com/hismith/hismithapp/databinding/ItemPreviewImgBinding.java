package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemPreviewImgBinding implements ViewBinding {
    public final ImageView ivPic;
    public final ImageView ivSmall;
    public final ProgressBar progressView;
    private final RelativeLayout rootView;
    public final View vCover;

    private ItemPreviewImgBinding(RelativeLayout relativeLayout, ImageView imageView, ImageView imageView2, ProgressBar progressBar, View view) {
        this.rootView = relativeLayout;
        this.ivPic = imageView;
        this.ivSmall = imageView2;
        this.progressView = progressBar;
        this.vCover = view;
    }

    public static ItemPreviewImgBinding bind(View view) {
        View findChildViewById;
        int i = R.id.iv_pic;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.iv_small;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.progressView;
                ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                if (progressBar != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.v_cover))) != null) {
                    return new ItemPreviewImgBinding((RelativeLayout) view, imageView, imageView2, progressBar, findChildViewById);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemPreviewImgBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPreviewImgBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_preview_img, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public RelativeLayout getRoot() {
        return this.rootView;
    }
}
