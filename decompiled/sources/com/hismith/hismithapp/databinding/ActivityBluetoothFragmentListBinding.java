package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.viewbinding.ViewBinding;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityBluetoothFragmentListBinding implements ViewBinding {
    private final LinearLayout rootView;

    private ActivityBluetoothFragmentListBinding(LinearLayout linearLayout) {
        this.rootView = linearLayout;
    }

    public static ActivityBluetoothFragmentListBinding bind(View view) {
        if (view != null) {
            return new ActivityBluetoothFragmentListBinding((LinearLayout) view);
        }
        throw new NullPointerException("rootView");
    }

    public static ActivityBluetoothFragmentListBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityBluetoothFragmentListBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_bluetooth_fragment_list, viewGroup, false);
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
