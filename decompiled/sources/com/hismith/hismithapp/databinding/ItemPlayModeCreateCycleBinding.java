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
public final class ItemPlayModeCreateCycleBinding implements ViewBinding {
    public final ImageView itemModeCreateImgType;
    public final FrameLayout itemModeCreateLayoutDelete;
    public final LinearLayout itemModeCreateLayoutStart;
    public final LinearLayout itemModeCreateLayoutStop;
    public final LinearLayout itemModeCreateLayoutTime;
    public final TextView itemModeCreateTxtEnd;
    public final TextView itemModeCreateTxtEndPhase;
    public final TextView itemModeCreateTxtStart;
    public final TextView itemModeCreateTxtStartPhase;
    public final TextView itemModeCreateTxtTime;
    public final TextView itemModeCreateTxtTimeValue;
    public final LinearLayout itemModeLayout;
    private final LinearLayout rootView;

    private ItemPlayModeCreateCycleBinding(LinearLayout linearLayout, ImageView imageView, FrameLayout frameLayout, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, LinearLayout linearLayout5) {
        this.rootView = linearLayout;
        this.itemModeCreateImgType = imageView;
        this.itemModeCreateLayoutDelete = frameLayout;
        this.itemModeCreateLayoutStart = linearLayout2;
        this.itemModeCreateLayoutStop = linearLayout3;
        this.itemModeCreateLayoutTime = linearLayout4;
        this.itemModeCreateTxtEnd = textView;
        this.itemModeCreateTxtEndPhase = textView2;
        this.itemModeCreateTxtStart = textView3;
        this.itemModeCreateTxtStartPhase = textView4;
        this.itemModeCreateTxtTime = textView5;
        this.itemModeCreateTxtTimeValue = textView6;
        this.itemModeLayout = linearLayout5;
    }

    public static ItemPlayModeCreateCycleBinding bind(View view) {
        int i = R.id.item_mode_create_img_type;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_mode_create_layout_delete;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.item_mode_create_layout_start;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.item_mode_create_layout_stop;
                    LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout2 != null) {
                        i = R.id.item_mode_create_layout_time;
                        LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout3 != null) {
                            i = R.id.item_mode_create_txt_end;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.item_mode_create_txt_end_phase;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    i = R.id.item_mode_create_txt_start;
                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView3 != null) {
                                        i = R.id.item_mode_create_txt_start_phase;
                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView4 != null) {
                                            i = R.id.item_mode_create_txt_time;
                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView5 != null) {
                                                i = R.id.item_mode_create_txt_time_value;
                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView6 != null) {
                                                    i = R.id.item_mode_layout;
                                                    LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                    if (linearLayout4 != null) {
                                                        return new ItemPlayModeCreateCycleBinding((LinearLayout) view, imageView, frameLayout, linearLayout, linearLayout2, linearLayout3, textView, textView2, textView3, textView4, textView5, textView6, linearLayout4);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemPlayModeCreateCycleBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlayModeCreateCycleBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_mode_create_cycle, viewGroup, false);
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
