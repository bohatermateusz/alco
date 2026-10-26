package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemMessageBinding implements ViewBinding {
    public final CircleImageView messageImgHead;
    public final TextView messageTxtContent;
    public final TextView messageTxtCount;
    public final TextView messageTxtName;
    public final TextView messageTxtTime;
    private final LinearLayout rootView;

    private ItemMessageBinding(LinearLayout linearLayout, CircleImageView circleImageView, TextView textView, TextView textView2, TextView textView3, TextView textView4) {
        this.rootView = linearLayout;
        this.messageImgHead = circleImageView;
        this.messageTxtContent = textView;
        this.messageTxtCount = textView2;
        this.messageTxtName = textView3;
        this.messageTxtTime = textView4;
    }

    public static ItemMessageBinding bind(View view) {
        int i = R.id.message_img_head;
        CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
        if (circleImageView != null) {
            i = R.id.message_txt_content;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.message_txt_count;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.message_txt_name;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        i = R.id.message_txt_time;
                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView4 != null) {
                            return new ItemMessageBinding((LinearLayout) view, circleImageView, textView, textView2, textView3, textView4);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemMessageBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemMessageBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_message, viewGroup, false);
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
