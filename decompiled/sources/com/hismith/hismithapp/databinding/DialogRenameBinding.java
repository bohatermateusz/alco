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
public final class DialogRenameBinding implements ViewBinding {
    public final EditText renameEditName;
    public final TextView renameTxtCancel;
    public final TextView renameTxtConfirm;
    public final TextView renameTxtTitle;
    private final LinearLayout rootView;

    private DialogRenameBinding(LinearLayout linearLayout, EditText editText, TextView textView, TextView textView2, TextView textView3) {
        this.rootView = linearLayout;
        this.renameEditName = editText;
        this.renameTxtCancel = textView;
        this.renameTxtConfirm = textView2;
        this.renameTxtTitle = textView3;
    }

    public static DialogRenameBinding bind(View view) {
        int i = R.id.rename_edit_name;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.rename_txt_cancel;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.rename_txt_confirm;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.rename_txt_title;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        return new DialogRenameBinding((LinearLayout) view, editText, textView, textView2, textView3);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogRenameBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogRenameBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_rename, viewGroup, false);
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
