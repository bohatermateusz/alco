package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemRemoteChatLeftBinding implements ViewBinding {
    public final FrameLayout chatItemLayout;
    public final TextView chatItemTxtContent;
    private final ConstraintLayout rootView;

    private ItemRemoteChatLeftBinding(ConstraintLayout constraintLayout, FrameLayout frameLayout, TextView textView) {
        this.rootView = constraintLayout;
        this.chatItemLayout = frameLayout;
        this.chatItemTxtContent = textView;
    }

    public static ItemRemoteChatLeftBinding bind(View view) {
        int i = R.id.chat_item_layout;
        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
        if (frameLayout != null) {
            i = R.id.chat_item_txt_content;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                return new ItemRemoteChatLeftBinding((ConstraintLayout) view, frameLayout, textView);
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemRemoteChatLeftBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemRemoteChatLeftBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_remote_chat_left, viewGroup, false);
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
