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
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemInteractionBinding implements ViewBinding {
    public final TextView itemInteractionBtnFollow;
    public final ImageView itemInteractionImgGender;
    public final CircleImageView itemInteractionImgHead;
    public final ImageView itemInteractionImgProgress;
    public final FrameLayout itemInteractionLayoutFollow;
    public final View itemInteractionLine;
    public final TextView itemInteractionTxtDate;
    public final TextView itemInteractionTxtLast;
    public final TextView itemInteractionTxtName;
    public final TextView itemInteractionTxtSourceContent;
    public final TextView itemInteractionTxtTypeNote;
    private final LinearLayout rootView;

    private ItemInteractionBinding(LinearLayout linearLayout, TextView textView, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, FrameLayout frameLayout, View view, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6) {
        this.rootView = linearLayout;
        this.itemInteractionBtnFollow = textView;
        this.itemInteractionImgGender = imageView;
        this.itemInteractionImgHead = circleImageView;
        this.itemInteractionImgProgress = imageView2;
        this.itemInteractionLayoutFollow = frameLayout;
        this.itemInteractionLine = view;
        this.itemInteractionTxtDate = textView2;
        this.itemInteractionTxtLast = textView3;
        this.itemInteractionTxtName = textView4;
        this.itemInteractionTxtSourceContent = textView5;
        this.itemInteractionTxtTypeNote = textView6;
    }

    public static ItemInteractionBinding bind(View view) {
        View findChildViewById;
        int i = R.id.item_interaction_btn_follow;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.item_interaction_img_gender;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.item_interaction_img_head;
                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                if (circleImageView != null) {
                    i = R.id.item_interaction_img_progress;
                    ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView2 != null) {
                        i = R.id.item_interaction_layout_follow;
                        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                        if (frameLayout != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.item_interaction_line))) != null) {
                            i = R.id.item_interaction_txt_date;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.item_interaction_txt_last;
                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView3 != null) {
                                    i = R.id.item_interaction_txt_name;
                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView4 != null) {
                                        i = R.id.item_interaction_txt_sourceContent;
                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView5 != null) {
                                            i = R.id.item_interaction_txt_typeNote;
                                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView6 != null) {
                                                return new ItemInteractionBinding((LinearLayout) view, textView, imageView, circleImageView, imageView2, frameLayout, findChildViewById, textView2, textView3, textView4, textView5, textView6);
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

    public static ItemInteractionBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemInteractionBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_interaction, viewGroup, false);
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
