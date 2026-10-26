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
public final class PopupChatBinding implements ViewBinding {
    public final View lineDelete;
    public final View lineReference;
    public final View lineRevoke;
    private final LinearLayout rootView;
    public final TextView txtCopy;
    public final TextView txtDelete;
    public final TextView txtReference;
    public final TextView txtRevoke;

    private PopupChatBinding(LinearLayout linearLayout, View view, View view2, View view3, TextView textView, TextView textView2, TextView textView3, TextView textView4) {
        this.rootView = linearLayout;
        this.lineDelete = view;
        this.lineReference = view2;
        this.lineRevoke = view3;
        this.txtCopy = textView;
        this.txtDelete = textView2;
        this.txtReference = textView3;
        this.txtRevoke = textView4;
    }

    public static PopupChatBinding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        int i = R.id.line_delete;
        View findChildViewById3 = ViewBindings.findChildViewById(view, i);
        if (findChildViewById3 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line_reference))) != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.line_revoke))) != null) {
            i = R.id.txt_copy;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.txt_delete;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.txt_reference;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        i = R.id.txt_revoke;
                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView4 != null) {
                            return new PopupChatBinding((LinearLayout) view, findChildViewById3, findChildViewById, findChildViewById2, textView, textView2, textView3, textView4);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static PopupChatBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PopupChatBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.popup_chat, viewGroup, false);
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
