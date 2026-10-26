package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatCheckBox;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ViewHoleMaskTipBinding implements ViewBinding {
    public final TextView btnHoleMaskTipGotIt;
    public final AppCompatCheckBox chkHoleMaskTipNoMore;
    public final View holeMaskTipArrow;
    public final LinearLayout layoutHoleMaskTipBubble;
    public final LinearLayout layoutHoleMaskTipContent;
    private final LinearLayout rootView;
    public final TextView txtHoleMaskTip;

    private ViewHoleMaskTipBinding(LinearLayout linearLayout, TextView textView, AppCompatCheckBox appCompatCheckBox, View view, LinearLayout linearLayout2, LinearLayout linearLayout3, TextView textView2) {
        this.rootView = linearLayout;
        this.btnHoleMaskTipGotIt = textView;
        this.chkHoleMaskTipNoMore = appCompatCheckBox;
        this.holeMaskTipArrow = view;
        this.layoutHoleMaskTipBubble = linearLayout2;
        this.layoutHoleMaskTipContent = linearLayout3;
        this.txtHoleMaskTip = textView2;
    }

    public static ViewHoleMaskTipBinding bind(View view) {
        View findChildViewById;
        int i = R.id.btn_hole_mask_tip_got_it;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.chk_hole_mask_tip_no_more;
            AppCompatCheckBox appCompatCheckBox = (AppCompatCheckBox) ViewBindings.findChildViewById(view, i);
            if (appCompatCheckBox != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.hole_mask_tip_arrow))) != null) {
                i = R.id.layout_hole_mask_tip_bubble;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    LinearLayout linearLayout2 = (LinearLayout) view;
                    i = R.id.txt_hole_mask_tip;
                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView2 != null) {
                        return new ViewHoleMaskTipBinding(linearLayout2, textView, appCompatCheckBox, findChildViewById, linearLayout, linearLayout2, textView2);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ViewHoleMaskTipBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ViewHoleMaskTipBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.view_hole_mask_tip, viewGroup, false);
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
