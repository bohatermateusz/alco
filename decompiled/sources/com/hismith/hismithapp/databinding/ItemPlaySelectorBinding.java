package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.Space;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemPlaySelectorBinding implements ViewBinding {
    public final ImageView imgPlayIcon;
    public final ConstraintLayout layoutItem;
    public final Guideline lineEnd;
    private final ConstraintLayout rootView;
    public final Space spaceOffset;
    public final AppCompatTextView txtPlayName;

    private ItemPlaySelectorBinding(ConstraintLayout constraintLayout, ImageView imageView, ConstraintLayout constraintLayout2, Guideline guideline, Space space, AppCompatTextView appCompatTextView) {
        this.rootView = constraintLayout;
        this.imgPlayIcon = imageView;
        this.layoutItem = constraintLayout2;
        this.lineEnd = guideline;
        this.spaceOffset = space;
        this.txtPlayName = appCompatTextView;
    }

    public static ItemPlaySelectorBinding bind(View view) {
        int i = R.id.img_play_icon;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.layout_item;
            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
            if (constraintLayout != null) {
                i = R.id.line_end;
                Guideline guideline = (Guideline) ViewBindings.findChildViewById(view, i);
                if (guideline != null) {
                    i = R.id.space_offset;
                    Space space = (Space) ViewBindings.findChildViewById(view, i);
                    if (space != null) {
                        i = R.id.txt_play_name;
                        AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                        if (appCompatTextView != null) {
                            return new ItemPlaySelectorBinding((ConstraintLayout) view, imageView, constraintLayout, guideline, space, appCompatTextView);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemPlaySelectorBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlaySelectorBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_selector, viewGroup, false);
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
