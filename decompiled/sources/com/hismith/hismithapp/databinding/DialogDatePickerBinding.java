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
public final class DialogDatePickerBinding implements ViewBinding {
    public final QNumberPicker dialogPickerDay;
    public final QNumberPicker dialogPickerMonth;
    public final TextView dialogPickerTxtSubmit;
    public final QNumberPicker dialogPickerYear;
    public final TextView reportSelectTxtTitle;
    private final LinearLayout rootView;

    private DialogDatePickerBinding(LinearLayout linearLayout, QNumberPicker qNumberPicker, QNumberPicker qNumberPicker2, TextView textView, QNumberPicker qNumberPicker3, TextView textView2) {
        this.rootView = linearLayout;
        this.dialogPickerDay = qNumberPicker;
        this.dialogPickerMonth = qNumberPicker2;
        this.dialogPickerTxtSubmit = textView;
        this.dialogPickerYear = qNumberPicker3;
        this.reportSelectTxtTitle = textView2;
    }

    public static DialogDatePickerBinding bind(View view) {
        int i = R.id.dialog_picker_day;
        QNumberPicker qNumberPicker = (QNumberPicker) ViewBindings.findChildViewById(view, i);
        if (qNumberPicker != null) {
            i = R.id.dialog_picker_month;
            QNumberPicker qNumberPicker2 = (QNumberPicker) ViewBindings.findChildViewById(view, i);
            if (qNumberPicker2 != null) {
                i = R.id.dialog_picker_txt_submit;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    i = R.id.dialog_picker_year;
                    QNumberPicker qNumberPicker3 = (QNumberPicker) ViewBindings.findChildViewById(view, i);
                    if (qNumberPicker3 != null) {
                        i = R.id.report_select_txt_title;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            return new DialogDatePickerBinding((LinearLayout) view, qNumberPicker, qNumberPicker2, textView, qNumberPicker3, textView2);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogDatePickerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogDatePickerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_date_picker, viewGroup, false);
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
