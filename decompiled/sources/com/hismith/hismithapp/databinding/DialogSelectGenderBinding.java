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
import com.hismith.hismithapp.view.WheelView;

/* loaded from: classes3.dex */
public final class DialogSelectGenderBinding implements ViewBinding {
    public final Button dialogGenderBtnCancel;
    public final Button dialogGenderBtnConfirm;
    public final TextView dialogGenderTxtTitle;
    public final WheelView dialogGenderWheel;
    private final LinearLayout rootView;

    private DialogSelectGenderBinding(LinearLayout linearLayout, Button button, Button button2, TextView textView, WheelView wheelView) {
        this.rootView = linearLayout;
        this.dialogGenderBtnCancel = button;
        this.dialogGenderBtnConfirm = button2;
        this.dialogGenderTxtTitle = textView;
        this.dialogGenderWheel = wheelView;
    }

    public static DialogSelectGenderBinding bind(View view) {
        int i = R.id.dialog_gender_btn_cancel;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.dialog_gender_btn_confirm;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.dialog_gender_txt_title;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    i = R.id.dialog_gender_wheel;
                    WheelView wheelView = (WheelView) ViewBindings.findChildViewById(view, i);
                    if (wheelView != null) {
                        return new DialogSelectGenderBinding((LinearLayout) view, button, button2, textView, wheelView);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogSelectGenderBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogSelectGenderBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_select_gender, viewGroup, false);
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
