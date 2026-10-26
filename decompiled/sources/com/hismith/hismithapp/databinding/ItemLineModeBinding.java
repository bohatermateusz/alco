package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemLineModeBinding implements ViewBinding {
    public final LinearLayout lineModeLayout;
    public final View lineModeLineBottom;
    public final View lineModeLineTop;
    public final TextView lineModeTxtName;
    public final TextView lineModeTxtTime;
    private final LinearLayout rootView;

    private ItemLineModeBinding(LinearLayout linearLayout, LinearLayout linearLayout2, View view, View view2, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.lineModeLayout = linearLayout2;
        this.lineModeLineBottom = view;
        this.lineModeLineTop = view2;
        this.lineModeTxtName = textView;
        this.lineModeTxtTime = textView2;
    }

    public static ItemLineModeBinding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        int i = R.id.line_mode_layout;
        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
        if (linearLayout != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line_mode_line_bottom))) != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.line_mode_line_top))) != null) {
            i = R.id.line_mode_txt_name;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.line_mode_txt_time;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    return new ItemLineModeBinding((LinearLayout) view, linearLayout, findChildViewById, findChildViewById2, textView, textView2);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemLineModeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemLineModeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_line_mode, viewGroup, false);
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
