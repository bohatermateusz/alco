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

/* loaded from: classes3.dex */
public final class ItemPlayModePopupBinding implements ViewBinding {
    public final ImageView popupImg;
    public final LinearLayout popupLayout;
    public final View popupLine;
    public final TextView popupTxt;
    private final LinearLayout rootView;

    private ItemPlayModePopupBinding(LinearLayout linearLayout, ImageView imageView, LinearLayout linearLayout2, View view, TextView textView) {
        this.rootView = linearLayout;
        this.popupImg = imageView;
        this.popupLayout = linearLayout2;
        this.popupLine = view;
        this.popupTxt = textView;
    }

    public static ItemPlayModePopupBinding bind(View view) {
        View findChildViewById;
        int i = R.id.popup_img;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.popup_layout;
            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
            if (linearLayout != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.popup_line))) != null) {
                i = R.id.popup_txt;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    return new ItemPlayModePopupBinding((LinearLayout) view, imageView, linearLayout, findChildViewById, textView);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemPlayModePopupBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlayModePopupBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_mode_popup, viewGroup, false);
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
