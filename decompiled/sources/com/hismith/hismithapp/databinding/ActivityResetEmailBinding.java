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
public final class ActivityResetEmailBinding implements ViewBinding {
    public final Button resetEmailBtnComplete;
    public final Button resetEmailBtnGetcode;
    public final EditText resetEmailEditNewEmailValue;
    public final EditText resetEmailEditVerificationCode;
    public final TextView resetEmailTxtEmail;
    public final TextView resetEmailTxtEmailValue;
    public final TextView resetEmailTxtNewEmail;
    public final TextView resetEmailTxtVerificationCode;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityResetEmailBinding(LinearLayout linearLayout, Button button, Button button2, EditText editText, EditText editText2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = linearLayout;
        this.resetEmailBtnComplete = button;
        this.resetEmailBtnGetcode = button2;
        this.resetEmailEditNewEmailValue = editText;
        this.resetEmailEditVerificationCode = editText2;
        this.resetEmailTxtEmail = textView;
        this.resetEmailTxtEmailValue = textView2;
        this.resetEmailTxtNewEmail = textView3;
        this.resetEmailTxtVerificationCode = textView4;
        this.titleTxtName = textView5;
    }

    public static ActivityResetEmailBinding bind(View view) {
        int i = R.id.reset_email_btn_complete;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.reset_email_btn_getcode;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.reset_email_edit_new_emailValue;
                EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                if (editText != null) {
                    i = R.id.reset_email_edit_verificationCode;
                    EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText2 != null) {
                        i = R.id.reset_email_txt_email;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.reset_email_txt_emailValue;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.reset_email_txt_new_email;
                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView3 != null) {
                                    i = R.id.reset_email_txt_verificationCode;
                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView4 != null) {
                                        i = R.id.title_txt_name;
                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView5 != null) {
                                            return new ActivityResetEmailBinding((LinearLayout) view, button, button2, editText, editText2, textView, textView2, textView3, textView4, textView5);
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

    public static ActivityResetEmailBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityResetEmailBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_reset_email, viewGroup, false);
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
