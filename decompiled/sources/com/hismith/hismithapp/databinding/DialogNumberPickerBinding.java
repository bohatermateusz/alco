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
public final class DialogNumberPickerBinding implements ViewBinding {
    public final QNumberPicker dialogPickerDot;
    public final QNumberPicker dialogPickerMilli;
    public final QNumberPicker dialogPickerSeconds;
    public final TextView dialogTxtSubmit;
    public final TextView dialogTxtTitle;
    private final LinearLayout rootView;

    private DialogNumberPickerBinding(LinearLayout linearLayout, QNumberPicker qNumberPicker, QNumberPicker qNumberPicker2, QNumberPicker qNumberPicker3, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.dialogPickerDot = qNumberPicker;
        this.dialogPickerMilli = qNumberPicker2;
        this.dialogPickerSeconds = qNumberPicker3;
        this.dialogTxtSubmit = textView;
        this.dialogTxtTitle = textView2;
    }

    public static DialogNumberPickerBinding bind(View view) {
        int i = R.id.dialog_picker_dot;
        QNumberPicker qNumberPicker = (QNumberPicker) ViewBindings.findChildViewById(view, i);
        if (qNumberPicker != null) {
            i = R.id.dialog_picker_milli;
            QNumberPicker qNumberPicker2 = (QNumberPicker) ViewBindings.findChildViewById(view, i);
            if (qNumberPicker2 != null) {
                i = R.id.dialog_picker_seconds;
                QNumberPicker qNumberPicker3 = (QNumberPicker) ViewBindings.findChildViewById(view, i);
                if (qNumberPicker3 != null) {
                    i = R.id.dialog_txt_submit;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.dialog_txt_title;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            return new DialogNumberPickerBinding((LinearLayout) view, qNumberPicker, qNumberPicker2, qNumberPicker3, textView, textView2);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogNumberPickerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogNumberPickerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_number_picker, viewGroup, false);
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
