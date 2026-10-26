package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemHudongSmsModeBinding implements ViewBinding {
    public final ImageView itemModeBg;
    public final ImageView itemModeImgMode;
    public final ImageView itemModeLayoutMode;
    private final ConstraintLayout rootView;
    public final View view;

    private ItemHudongSmsModeBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, View view) {
        this.rootView = constraintLayout;
        this.itemModeBg = imageView;
        this.itemModeImgMode = imageView2;
        this.itemModeLayoutMode = imageView3;
        this.view = view;
    }

    public static ItemHudongSmsModeBinding bind(View view) {
        View findChildViewById;
        int i = R.id.item_mode_bg;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_mode_img_mode;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.item_mode_layout_mode;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view))) != null) {
                    return new ItemHudongSmsModeBinding((ConstraintLayout) view, imageView, imageView2, imageView3, findChildViewById);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemHudongSmsModeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemHudongSmsModeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_hudong_sms_mode, viewGroup, false);
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
