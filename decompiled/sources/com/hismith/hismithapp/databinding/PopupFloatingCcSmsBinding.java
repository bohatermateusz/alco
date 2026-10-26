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
public final class PopupFloatingCcSmsBinding implements ViewBinding {
    public final ImageView popupFloatImgBg;
    public final ImageView popupFloatImgMode;
    public final FrameLayout popupFloatLayout;
    public final ImageView popupFloatModeBg;
    public final TextView popupFloatTxtSpeed;
    public final TextView popupTxtAuto;
    public final View popupViewMode;
    private final FrameLayout rootView;

    private PopupFloatingCcSmsBinding(FrameLayout frameLayout, ImageView imageView, ImageView imageView2, FrameLayout frameLayout2, ImageView imageView3, TextView textView, TextView textView2, View view) {
        this.rootView = frameLayout;
        this.popupFloatImgBg = imageView;
        this.popupFloatImgMode = imageView2;
        this.popupFloatLayout = frameLayout2;
        this.popupFloatModeBg = imageView3;
        this.popupFloatTxtSpeed = textView;
        this.popupTxtAuto = textView2;
        this.popupViewMode = view;
    }

    public static PopupFloatingCcSmsBinding bind(View view) {
        View findChildViewById;
        int i = R.id.popup_float_img_bg;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.popup_float_img_mode;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.popup_float_layout;
                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout != null) {
                    i = R.id.popup_float_mode_bg;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.popup_float_txt_speed;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.popup_txt_auto;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.popup_view_mode))) != null) {
                                return new PopupFloatingCcSmsBinding((FrameLayout) view, imageView, imageView2, frameLayout, imageView3, textView, textView2, findChildViewById);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static PopupFloatingCcSmsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PopupFloatingCcSmsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.popup_floating_cc_sms, viewGroup, false);
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
