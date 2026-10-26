package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.SlideRecyclerView;

/* loaded from: classes3.dex */
public final class DialogDrawerUserBinding implements ViewBinding {
    public final SlideRecyclerView dialogDrawerRecyclerUser;
    private final RelativeLayout rootView;
    public final LinearLayout titleLayout;
    public final TextView titleTxtName;

    private DialogDrawerUserBinding(RelativeLayout relativeLayout, SlideRecyclerView slideRecyclerView, LinearLayout linearLayout, TextView textView) {
        this.rootView = relativeLayout;
        this.dialogDrawerRecyclerUser = slideRecyclerView;
        this.titleLayout = linearLayout;
        this.titleTxtName = textView;
    }

    public static DialogDrawerUserBinding bind(View view) {
        int i = R.id.dialog_drawer_recycler_user;
        SlideRecyclerView slideRecyclerView = (SlideRecyclerView) ViewBindings.findChildViewById(view, i);
        if (slideRecyclerView != null) {
            i = R.id.title_layout;
            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
            if (linearLayout != null) {
                i = R.id.title_txt_name;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    return new DialogDrawerUserBinding((RelativeLayout) view, slideRecyclerView, linearLayout, textView);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogDrawerUserBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogDrawerUserBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_drawer_user, viewGroup, false);
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
