package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ShDefaultProgressLayoutBinding implements ViewBinding {
    private final RelativeLayout rootView;
    public final TextView shProgressText;
    public final ProgressBar shProgressView;

    private ShDefaultProgressLayoutBinding(RelativeLayout relativeLayout, TextView textView, ProgressBar progressBar) {
        this.rootView = relativeLayout;
        this.shProgressText = textView;
        this.shProgressView = progressBar;
    }

    public static ShDefaultProgressLayoutBinding bind(View view) {
        int i = R.id.sh_progress_text;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.sh_progress_view;
            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
            if (progressBar != null) {
                return new ShDefaultProgressLayoutBinding((RelativeLayout) view, textView, progressBar);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ShDefaultProgressLayoutBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ShDefaultProgressLayoutBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.sh_default_progress_layout, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public RelativeLayout getRoot() {
        return this.rootView;
    }
}
