package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class PlugSettingsBinding implements ViewBinding {
    private final ConstraintLayout rootView;

    private PlugSettingsBinding(ConstraintLayout constraintLayout) {
        this.rootView = constraintLayout;
    }

    public static PlugSettingsBinding bind(View view) {
        if (view != null) {
            return new PlugSettingsBinding((ConstraintLayout) view);
        }
        throw new NullPointerException("rootView");
    }

    public static PlugSettingsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PlugSettingsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.plug_settings, viewGroup, false);
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
