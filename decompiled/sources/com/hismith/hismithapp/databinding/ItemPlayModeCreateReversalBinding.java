package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemPlayModeCreateReversalBinding implements ViewBinding {
    public final ImageView itemModeCreateImgType;
    public final FrameLayout itemModeCreateLayoutDelete;
    public final LinearLayout itemModeCreateLayoutStop;
    public final TextView itemModeCreateTxtEnd;
    public final TextView itemModeCreateTxtEndValue;
    public final TextView itemModeCreateTxtSerial;
    public final LinearLayout itemModeLayout;
    private final LinearLayout rootView;

    private ItemPlayModeCreateReversalBinding(LinearLayout linearLayout, ImageView imageView, FrameLayout frameLayout, LinearLayout linearLayout2, TextView textView, TextView textView2, TextView textView3, LinearLayout linearLayout3) {
        this.rootView = linearLayout;
        this.itemModeCreateImgType = imageView;
        this.itemModeCreateLayoutDelete = frameLayout;
        this.itemModeCreateLayoutStop = linearLayout2;
        this.itemModeCreateTxtEnd = textView;
        this.itemModeCreateTxtEndValue = textView2;
        this.itemModeCreateTxtSerial = textView3;
        this.itemModeLayout = linearLayout3;
    }

    public static ItemPlayModeCreateReversalBinding bind(View view) {
        int i = R.id.item_mode_create_img_type;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_mode_create_layout_delete;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.item_mode_create_layout_stop;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.item_mode_create_txt_end;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.item_mode_create_txt_end_value;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            i = R.id.item_mode_create_txt_serial;
                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView3 != null) {
                                i = R.id.item_mode_layout;
                                LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout2 != null) {
                                    return new ItemPlayModeCreateReversalBinding((LinearLayout) view, imageView, frameLayout, linearLayout, textView, textView2, textView3, linearLayout2);
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemPlayModeCreateReversalBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlayModeCreateReversalBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_mode_create_reversal, viewGroup, false);
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
