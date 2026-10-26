package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class FragmentBluetoothMultitudeBinding implements ViewBinding {
    public final SlideRecyclerView recyclerBluetooth;
    public final SmartRefreshLayout refreshLayout;
    private final ConstraintLayout rootView;
    public final LinearLayout titleLayout;
    public final TextView titleTxtName;

    private FragmentBluetoothMultitudeBinding(ConstraintLayout constraintLayout, SlideRecyclerView slideRecyclerView, SmartRefreshLayout smartRefreshLayout, LinearLayout linearLayout, TextView textView) {
        this.rootView = constraintLayout;
        this.recyclerBluetooth = slideRecyclerView;
        this.refreshLayout = smartRefreshLayout;
        this.titleLayout = linearLayout;
        this.titleTxtName = textView;
    }

    public static FragmentBluetoothMultitudeBinding bind(View view) {
        int i = R.id.recycler_bluetooth;
        SlideRecyclerView slideRecyclerView = (SlideRecyclerView) ViewBindings.findChildViewById(view, i);
        if (slideRecyclerView != null) {
            i = R.id.refresh_layout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                i = R.id.title_layout;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.title_txt_name;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        return new FragmentBluetoothMultitudeBinding((ConstraintLayout) view, slideRecyclerView, smartRefreshLayout, linearLayout, textView);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentBluetoothMultitudeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentBluetoothMultitudeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_bluetooth_multitude, viewGroup, false);
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
