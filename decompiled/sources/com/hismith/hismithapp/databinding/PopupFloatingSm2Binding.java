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
public final class PopupFloatingSm2Binding implements ViewBinding {
    public final ImageView popupFloatImgBg;
    public final ImageView popupFloatImgMode;
    public final ImageView popupFloatImgVibrate;
    public final FrameLayout popupFloatLayout;
    public final ImageView popupFloatModeBg;
    public final TextView popupFloatTxtSpeed;
    public final View popupViewMode;
    public final View popupViewMusic;
    public final View popupViewVibrate;
    private final FrameLayout rootView;

    private PopupFloatingSm2Binding(FrameLayout frameLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, FrameLayout frameLayout2, ImageView imageView4, TextView textView, View view, View view2, View view3) {
        this.rootView = frameLayout;
        this.popupFloatImgBg = imageView;
        this.popupFloatImgMode = imageView2;
        this.popupFloatImgVibrate = imageView3;
        this.popupFloatLayout = frameLayout2;
        this.popupFloatModeBg = imageView4;
        this.popupFloatTxtSpeed = textView;
        this.popupViewMode = view;
        this.popupViewMusic = view2;
        this.popupViewVibrate = view3;
    }

    public static PopupFloatingSm2Binding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        View findChildViewById3;
        int i = R.id.popup_float_img_bg;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.popup_float_img_mode;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.popup_float_img_vibrate;
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
                            if (textView != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.popup_view_mode))) != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.popup_view_music))) != null && (findChildViewById3 = ViewBindings.findChildViewById(view, (i = R.id.popup_view_vibrate))) != null) {
                                return new PopupFloatingSm2Binding((FrameLayout) view, imageView, imageView2, imageView3, frameLayout, imageView4, textView, findChildViewById, findChildViewById2, findChildViewById3);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static PopupFloatingSm2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PopupFloatingSm2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.popup_floating_sm2, viewGroup, false);
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
