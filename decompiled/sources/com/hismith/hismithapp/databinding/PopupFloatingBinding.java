package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class PopupFloatingBinding implements ViewBinding {
    public final ImageView popupFloatImgBg;
    public final ImageView popupFloatImgHeart;
    public final ImageView popupFloatImgMode;
    public final FrameLayout popupFloatLayout;
    public final ImageView popupFloatModeBg;
    public final TextView popupFloatTxtSpeed;
    private final FrameLayout rootView;

    private PopupFloatingBinding(FrameLayout frameLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, FrameLayout frameLayout2, ImageView imageView4, TextView textView) {
        this.rootView = frameLayout;
        this.popupFloatImgBg = imageView;
        this.popupFloatImgHeart = imageView2;
        this.popupFloatImgMode = imageView3;
        this.popupFloatLayout = frameLayout2;
        this.popupFloatModeBg = imageView4;
        this.popupFloatTxtSpeed = textView;
    }

    public static PopupFloatingBinding bind(View view) {
        int i = R.id.popup_float_img_bg;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.popup_float_img_heart;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.popup_float_img_mode;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.popup_float_layout;
                    FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                    if (frameLayout != null) {
                        i = R.id.popup_float_mode_bg;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.popup_float_txt_speed;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                return new PopupFloatingBinding((FrameLayout) view, imageView, imageView2, imageView3, frameLayout, imageView4, textView);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static PopupFloatingBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PopupFloatingBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.popup_floating, viewGroup, false);
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
