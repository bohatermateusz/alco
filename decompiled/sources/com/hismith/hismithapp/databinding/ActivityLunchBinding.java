package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import pl.droidsonroids.gif.GifImageView;

/* loaded from: classes3.dex */
public final class ActivityLunchBinding implements ViewBinding {
    public final GifImageView gifLunch;
    private final LinearLayout rootView;

    private ActivityLunchBinding(LinearLayout linearLayout, GifImageView gifImageView) {
        this.rootView = linearLayout;
        this.gifLunch = gifImageView;
    }

    public static ActivityLunchBinding bind(View view) {
        int i = R.id.gif_lunch;
        GifImageView gifImageView = (GifImageView) ViewBindings.findChildViewById(view, i);
        if (gifImageView != null) {
            return new ActivityLunchBinding((LinearLayout) view, gifImageView);
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityLunchBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityLunchBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_lunch, viewGroup, false);
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
