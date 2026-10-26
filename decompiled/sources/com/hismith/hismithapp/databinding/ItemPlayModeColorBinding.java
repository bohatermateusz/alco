package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemPlayModeColorBinding implements ViewBinding {
    public final FrameLayout layoutColor;
    private final ConstraintLayout rootView;
    public final View viewColor;

    private ItemPlayModeColorBinding(ConstraintLayout constraintLayout, FrameLayout frameLayout, View view) {
        this.rootView = constraintLayout;
        this.layoutColor = frameLayout;
        this.viewColor = view;
    }

    public static ItemPlayModeColorBinding bind(View view) {
        View findChildViewById;
        int i = R.id.layout_color;
        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
        if (frameLayout == null || (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_color))) == null) {
            throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
        }
        return new ItemPlayModeColorBinding((ConstraintLayout) view, frameLayout, findChildViewById);
    }

    public static ItemPlayModeColorBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlayModeColorBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_mode_color, viewGroup, false);
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
