package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.viewbinding.ViewBinding;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogProgressBinding implements ViewBinding {
    private final LinearLayout rootView;

    private DialogProgressBinding(LinearLayout linearLayout) {
        this.rootView = linearLayout;
    }

    public static DialogProgressBinding bind(View view) {
        if (view != null) {
            return new DialogProgressBinding((LinearLayout) view);
        }
        throw new NullPointerException("rootView");
    }

    public static DialogProgressBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogProgressBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_progress, viewGroup, false);
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
