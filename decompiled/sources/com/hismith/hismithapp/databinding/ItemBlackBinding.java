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
public final class ItemBlackBinding implements ViewBinding {
    public final ImageView itemBlackImgGender;
    public final CircleImageView itemBlackImgHead;
    public final ImageView itemBlackImgSelected;
    public final TextView itemBlackTxtName;
    private final LinearLayout rootView;

    private ItemBlackBinding(LinearLayout linearLayout, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, TextView textView) {
        this.rootView = linearLayout;
        this.itemBlackImgGender = imageView;
        this.itemBlackImgHead = circleImageView;
        this.itemBlackImgSelected = imageView2;
        this.itemBlackTxtName = textView;
    }

    public static ItemBlackBinding bind(View view) {
        int i = R.id.item_black_img_gender;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_black_img_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.item_black_img_selected;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.item_black_txt_name;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        return new ItemBlackBinding((LinearLayout) view, imageView, circleImageView, imageView2, textView);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemBlackBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemBlackBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_black, viewGroup, false);
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
