package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class DialogLineUserBinding implements ViewBinding {
    public final Button btnCancel;
    public final Button btnComplete;
    public final LinearLayout layoutBtn;
    public final FrameLayout layoutTitle;
    public final RecyclerView recyclerView;
    public final SmartRefreshLayout refreshLayout;
    private final ConstraintLayout rootView;
    public final TextView txtTitle;

    private DialogLineUserBinding(ConstraintLayout constraintLayout, Button button, Button button2, LinearLayout linearLayout, FrameLayout frameLayout, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, TextView textView) {
        this.rootView = constraintLayout;
        this.btnCancel = button;
        this.btnComplete = button2;
        this.layoutBtn = linearLayout;
        this.layoutTitle = frameLayout;
        this.recyclerView = recyclerView;
        this.refreshLayout = smartRefreshLayout;
        this.txtTitle = textView;
    }

    public static DialogLineUserBinding bind(View view) {
        int i = R.id.btn_cancel;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.btn_complete;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.layout_btn;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.layout_title;
                    FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                    if (frameLayout != null) {
                        i = R.id.recycler_view;
                        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                        if (recyclerView != null) {
                            i = R.id.refreshLayout;
                            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                            if (smartRefreshLayout != null) {
                                i = R.id.txt_title;
                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView != null) {
                                    return new DialogLineUserBinding((ConstraintLayout) view, button, button2, linearLayout, frameLayout, recyclerView, smartRefreshLayout, textView);
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogLineUserBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogLineUserBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_line_user, viewGroup, false);
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
