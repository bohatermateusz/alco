package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityResetBinding implements ViewBinding {
    public final Button resetBtnComplete;
    public final EditText resetEditConfirmPassword;
    public final EditText resetEditNewPassword;
    public final EditText resetEditOriginalPassword;
    public final TextView resetTxtConfirmPassword;
    public final TextView resetTxtEmail;
    public final TextView resetTxtEmailValue;
    public final TextView resetTxtLabel;
    public final TextView resetTxtNewPassword;
    public final TextView resetTxtOriginalPassword;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityResetBinding(LinearLayout linearLayout, Button button, EditText editText, EditText editText2, EditText editText3, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7) {
        this.rootView = linearLayout;
        this.resetBtnComplete = button;
        this.resetEditConfirmPassword = editText;
        this.resetEditNewPassword = editText2;
        this.resetEditOriginalPassword = editText3;
        this.resetTxtConfirmPassword = textView;
        this.resetTxtEmail = textView2;
        this.resetTxtEmailValue = textView3;
        this.resetTxtLabel = textView4;
        this.resetTxtNewPassword = textView5;
        this.resetTxtOriginalPassword = textView6;
        this.titleTxtName = textView7;
    }

    public static ActivityResetBinding bind(View view) {
        int i = R.id.reset_btn_complete;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.reset_edit_confirmPassword;
            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText != null) {
                i = R.id.reset_edit_NewPassword;
                EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                if (editText2 != null) {
                    i = R.id.reset_edit_originalPassword;
                    EditText editText3 = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText3 != null) {
                        i = R.id.reset_txt_confirmPassword;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.reset_txt_email;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.reset_txt_emailValue;
                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView3 != null) {
                                    i = R.id.reset_txt_label;
                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView4 != null) {
                                        i = R.id.reset_txt_NewPassword;
                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView5 != null) {
                                            i = R.id.reset_txt_originalPassword;
                                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView6 != null) {
                                                i = R.id.title_txt_name;
                                                TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView7 != null) {
                                                    return new ActivityResetBinding((LinearLayout) view, button, editText, editText2, editText3, textView, textView2, textView3, textView4, textView5, textView6, textView7);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityResetBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityResetBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_reset, viewGroup, false);
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
