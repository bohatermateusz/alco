package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.QNumberPicker;

/* loaded from: classes3.dex */
public final class DialogCountPickerBinding implements ViewBinding {
    public final QNumberPicker dialogPickerCount;
    public final TextView dialogTxtSubmit;
    public final TextView dialogTxtTitle;
    private final LinearLayout rootView;

    private DialogCountPickerBinding(LinearLayout linearLayout, QNumberPicker qNumberPicker, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.dialogPickerCount = qNumberPicker;
        this.dialogTxtSubmit = textView;
        this.dialogTxtTitle = textView2;
    }

    public static DialogCountPickerBinding bind(View view) {
        int i = R.id.dialog_picker_count;
        QNumberPicker qNumberPicker = (QNumberPicker) ViewBindings.findChildViewById(view, i);
        if (qNumberPicker != null) {
            i = R.id.dialog_txt_submit;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.dialog_txt_title;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    return new DialogCountPickerBinding((LinearLayout) view, qNumberPicker, textView, textView2);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogCountPickerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogCountPickerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_count_picker, viewGroup, false);
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
