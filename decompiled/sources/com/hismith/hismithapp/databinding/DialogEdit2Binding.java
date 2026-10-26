package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogEdit2Binding implements ViewBinding {
    public final EditText dialogEditEditName;
    public final TextView dialogEditTxtCancel;
    public final TextView dialogEditTxtConfirm;
    public final TextView dialogEditTxtTitle;
    private final LinearLayout rootView;

    private DialogEdit2Binding(LinearLayout linearLayout, EditText editText, TextView textView, TextView textView2, TextView textView3) {
        this.rootView = linearLayout;
        this.dialogEditEditName = editText;
        this.dialogEditTxtCancel = textView;
        this.dialogEditTxtConfirm = textView2;
        this.dialogEditTxtTitle = textView3;
    }

    public static DialogEdit2Binding bind(View view) {
        int i = R.id.dialog_edit_edit_name;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.dialog_edit_txt_cancel;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.dialog_edit_txt_confirm;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.dialog_edit_txt_title;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        return new DialogEdit2Binding((LinearLayout) view, editText, textView, textView2, textView3);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogEdit2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogEdit2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_edit2, viewGroup, false);
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
