package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemSpinnerBinding implements ViewBinding {
    private final TextView rootView;

    private ItemSpinnerBinding(TextView textView) {
        this.rootView = textView;
    }

    public static ItemSpinnerBinding bind(View view) {
        if (view != null) {
            return new ItemSpinnerBinding((TextView) view);
        }
        throw new NullPointerException("rootView");
    }

    public static ItemSpinnerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemSpinnerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_spinner, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public TextView getRoot() {
        return this.rootView;
    }
}
