package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.slideingtab.MsgView;

/* loaded from: classes3.dex */
public final class LayoutTabBinding implements ViewBinding {
    private final RelativeLayout rootView;
    public final MsgView rtvMsgTip;
    public final TextView tvTabTitle;

    private LayoutTabBinding(RelativeLayout relativeLayout, MsgView msgView, TextView textView) {
        this.rootView = relativeLayout;
        this.rtvMsgTip = msgView;
        this.tvTabTitle = textView;
    }

    public static LayoutTabBinding bind(View view) {
        int i = R.id.rtv_msg_tip;
        MsgView msgView = (MsgView) ViewBindings.findChildViewById(view, i);
        if (msgView != null) {
            i = R.id.tv_tab_title;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                return new LayoutTabBinding((RelativeLayout) view, msgView, textView);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static LayoutTabBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static LayoutTabBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.layout_tab, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public RelativeLayout getRoot() {
        return this.rootView;
    }
}
