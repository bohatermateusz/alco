package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.LinePointsView;

/* loaded from: classes3.dex */
public final class AdapterModeListBinding implements ViewBinding {
    public final LinePointsView linePointsView;
    private final LinearLayout rootView;
    public final TextView txtModeName;
    public final TextView txtModeTime;

    private AdapterModeListBinding(LinearLayout linearLayout, LinePointsView linePointsView, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.linePointsView = linePointsView;
        this.txtModeName = textView;
        this.txtModeTime = textView2;
    }

    public static AdapterModeListBinding bind(View view) {
        int i = R.id.line_points_view;
        LinePointsView linePointsView = (LinePointsView) ViewBindings.findChildViewById(view, i);
        if (linePointsView != null) {
            i = R.id.txt_mode_name;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.txt_mode_time;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    return new AdapterModeListBinding((LinearLayout) view, linePointsView, textView, textView2);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static AdapterModeListBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static AdapterModeListBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.adapter_mode_list, viewGroup, false);
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
