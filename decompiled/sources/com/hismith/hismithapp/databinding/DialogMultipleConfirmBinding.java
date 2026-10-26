package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogMultipleConfirmBinding implements ViewBinding {
    public final Button dialogConfirmBtnCancel;
    public final Button dialogConfirmBtnConfirm;
    public final Button dialogConfirmBtnConfirm1;
    public final LinearLayout dialogConfirmLayoutContent;
    public final LinearLayout dialogConfirmLayoutTitle;
    public final TextView dialogConfirmTxtContent;
    public final TextView dialogConfirmTxtTitle;
    private final LinearLayout rootView;

    private DialogMultipleConfirmBinding(LinearLayout linearLayout, Button button, Button button2, Button button3, LinearLayout linearLayout2, LinearLayout linearLayout3, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.dialogConfirmBtnCancel = button;
        this.dialogConfirmBtnConfirm = button2;
        this.dialogConfirmBtnConfirm1 = button3;
        this.dialogConfirmLayoutContent = linearLayout2;
        this.dialogConfirmLayoutTitle = linearLayout3;
        this.dialogConfirmTxtContent = textView;
        this.dialogConfirmTxtTitle = textView2;
    }

    public static DialogMultipleConfirmBinding bind(View view) {
        int i = R.id.dialog_confirm_btn_cancel;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.dialog_confirm_btn_confirm;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.dialog_confirm_btn_confirm_1;
                Button button3 = (Button) ViewBindings.findChildViewById(view, i);
                if (button3 != null) {
                    i = R.id.dialog_confirm_layout_content;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.dialog_confirm_layout_title;
                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout2 != null) {
                            i = R.id.dialog_confirm_txt_content;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.dialog_confirm_txt_title;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    return new DialogMultipleConfirmBinding((LinearLayout) view, button, button2, button3, linearLayout, linearLayout2, textView, textView2);
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogMultipleConfirmBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogMultipleConfirmBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_multiple_confirm, viewGroup, false);
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
