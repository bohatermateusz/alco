package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemEditImageBinding implements ViewBinding {
    public final ImageView itemTopicGridImage;
    public final ImageView itemTopicGridNormal;
    public final ProgressBar itemTopicGridProgress;
    private final FrameLayout rootView;

    private ItemEditImageBinding(FrameLayout frameLayout, ImageView imageView, ImageView imageView2, ProgressBar progressBar) {
        this.rootView = frameLayout;
        this.itemTopicGridImage = imageView;
        this.itemTopicGridNormal = imageView2;
        this.itemTopicGridProgress = progressBar;
    }

    public static ItemEditImageBinding bind(View view) {
        int i = R.id.item_topic_grid_image;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_topic_grid_normal;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.item_topic_grid_progress;
                ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                if (progressBar != null) {
                    return new ItemEditImageBinding((FrameLayout) view, imageView, imageView2, progressBar);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemEditImageBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemEditImageBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_edit_image, viewGroup, false);
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
