package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivitySettingBinding implements ViewBinding {
    public final TitleBarBinding include;
    private final LinearLayout rootView;

    private ActivitySettingBinding(LinearLayout linearLayout, TitleBarBinding titleBarBinding) {
        this.rootView = linearLayout;
        this.include = titleBarBinding;
    }

    public static ActivitySettingBinding bind(View view) {
        int i = R.id.include;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById == null) {
            throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
        }
        return new ActivitySettingBinding((LinearLayout) view, TitleBarBinding.bind(findChildViewById));
    }

    public static ActivitySettingBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivitySettingBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_setting, viewGroup, false);
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
