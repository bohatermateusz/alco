package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemLikeUserBinding implements ViewBinding {
    public final ImageView fansImgGender;
    public final CircleImageView fansImgHead;
    public final TextView fansTxtName;
    private final LinearLayout rootView;

    private ItemLikeUserBinding(LinearLayout linearLayout, ImageView imageView, CircleImageView circleImageView, TextView textView) {
        this.rootView = linearLayout;
        this.fansImgGender = imageView;
        this.fansImgHead = circleImageView;
        this.fansTxtName = textView;
    }

    public static ItemLikeUserBinding bind(View view) {
        int i = R.id.fans_img_gender;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.fans_img_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.fans_txt_name;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    return new ItemLikeUserBinding((LinearLayout) view, imageView, circleImageView, textView);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemLikeUserBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemLikeUserBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_like_user, viewGroup, false);
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
