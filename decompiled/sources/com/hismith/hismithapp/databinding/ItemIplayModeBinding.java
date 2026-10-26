package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemIplayModeBinding implements ViewBinding {
    public final ImageView itemModeBg;
    public final ImageView itemModeImgMode;
    public final ImageView itemModeLayoutMode;
    public final ProgressBar itemModeProgressMode;
    private final ConstraintLayout rootView;

    private ItemIplayModeBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ProgressBar progressBar) {
        this.rootView = constraintLayout;
        this.itemModeBg = imageView;
        this.itemModeImgMode = imageView2;
        this.itemModeLayoutMode = imageView3;
        this.itemModeProgressMode = progressBar;
    }

    public static ItemIplayModeBinding bind(View view) {
        int i = R.id.item_mode_bg;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_mode_img_mode;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.item_mode_layout_mode;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.item_mode_progress_mode;
                    ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                    if (progressBar != null) {
                        return new ItemIplayModeBinding((ConstraintLayout) view, imageView, imageView2, imageView3, progressBar);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemIplayModeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemIplayModeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_iplay_mode, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public ConstraintLayout getRoot() {
        return this.rootView;
    }
}
