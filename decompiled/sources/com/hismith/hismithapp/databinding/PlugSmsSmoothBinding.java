package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomSeekbar;

/* loaded from: classes3.dex */
public final class PlugSmsSmoothBinding implements ViewBinding {
    private final ConstraintLayout rootView;
    public final CustomSeekbar seekSmooth;

    private PlugSmsSmoothBinding(ConstraintLayout constraintLayout, CustomSeekbar customSeekbar) {
        this.rootView = constraintLayout;
        this.seekSmooth = customSeekbar;
    }

    public static PlugSmsSmoothBinding bind(View view) {
        int i = R.id.seek_smooth;
        CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
        if (customSeekbar != null) {
            return new PlugSmsSmoothBinding((ConstraintLayout) view, customSeekbar);
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static PlugSmsSmoothBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PlugSmsSmoothBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.plug_sms_smooth, viewGroup, false);
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
