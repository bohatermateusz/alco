package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogActionAttachListBinding implements ViewBinding {
    public final FrameLayout dialogLayout;
    public final RecyclerView dialogRecyclerView;
    private final FrameLayout rootView;

    private DialogActionAttachListBinding(FrameLayout frameLayout, FrameLayout frameLayout2, RecyclerView recyclerView) {
        this.rootView = frameLayout;
        this.dialogLayout = frameLayout2;
        this.dialogRecyclerView = recyclerView;
    }

    public static DialogActionAttachListBinding bind(View view) {
        FrameLayout frameLayout = (FrameLayout) view;
        int i = R.id.dialog_recycler_view;
        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
        if (recyclerView != null) {
            return new DialogActionAttachListBinding(frameLayout, frameLayout, recyclerView);
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogActionAttachListBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogActionAttachListBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_action_attach_list, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public FrameLayout getRoot() {
        return this.rootView;
    }
}
