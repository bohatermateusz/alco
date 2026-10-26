package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemCircleImgBinding implements ViewBinding {
    public final CircleImageView itemCircleImg;
    public final ImageView itemCircleImgCaptain;
    private final LinearLayout rootView;

    private ItemCircleImgBinding(LinearLayout linearLayout, CircleImageView circleImageView, ImageView imageView) {
        this.rootView = linearLayout;
        this.itemCircleImg = circleImageView;
        this.itemCircleImgCaptain = imageView;
    }

    public static ItemCircleImgBinding bind(View view) {
        int i = R.id.item_circle_img;
        CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
        if (circleImageView != null) {
            i = R.id.item_circle_img_captain;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                return new ItemCircleImgBinding((LinearLayout) view, circleImageView, imageView);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemCircleImgBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemCircleImgBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_circle_img, viewGroup, false);
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
