package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityBluetootListBinding implements ViewBinding {
    public final SlideRecyclerView bluetoothRecycler;
    public final SmartRefreshLayout bluetoothRefreshLayout;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityBluetootListBinding(LinearLayout linearLayout, SlideRecyclerView slideRecyclerView, SmartRefreshLayout smartRefreshLayout, TextView textView) {
        this.rootView = linearLayout;
        this.bluetoothRecycler = slideRecyclerView;
        this.bluetoothRefreshLayout = smartRefreshLayout;
        this.titleTxtName = textView;
    }

    public static ActivityBluetootListBinding bind(View view) {
        int i = R.id.bluetooth_recycler;
        SlideRecyclerView slideRecyclerView = (SlideRecyclerView) ViewBindings.findChildViewById(view, i);
        if (slideRecyclerView != null) {
            i = R.id.bluetooth_refreshLayout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                i = R.id.title_txt_name;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    return new ActivityBluetootListBinding((LinearLayout) view, slideRecyclerView, smartRefreshLayout, textView);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityBluetootListBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityBluetootListBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_bluetoot_list, viewGroup, false);
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
