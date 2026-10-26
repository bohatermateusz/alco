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
public final class ActivityForgotBinding implements ViewBinding {
    public final Button forgotBtnForgot;
    public final Button forgotBtnGetcode;
    public final EditText forgotEditCode;
    public final EditText forgotEditMail;
    public final EditText forgotEditPwd;
    public final EditText forgotEditRepwd;
    public final TextView forgotTxtCode;
    public final TextView forgotTxtMail;
    public final TextView forgotTxtPwd;
    public final TextView forgotTxtRepwd;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityForgotBinding(LinearLayout linearLayout, Button button, Button button2, EditText editText, EditText editText2, EditText editText3, EditText editText4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = linearLayout;
        this.forgotBtnForgot = button;
        this.forgotBtnGetcode = button2;
        this.forgotEditCode = editText;
        this.forgotEditMail = editText2;
        this.forgotEditPwd = editText3;
        this.forgotEditRepwd = editText4;
        this.forgotTxtCode = textView;
        this.forgotTxtMail = textView2;
        this.forgotTxtPwd = textView3;
        this.forgotTxtRepwd = textView4;
        this.titleTxtName = textView5;
    }

    public static ActivityForgotBinding bind(View view) {
        int i = R.id.forgot_btn_forgot;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.forgot_btn_getcode;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.forgot_edit_code;
                EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                if (editText != null) {
                    i = R.id.forgot_edit_mail;
                    EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText2 != null) {
                        i = R.id.forgot_edit_pwd;
                        EditText editText3 = (EditText) ViewBindings.findChildViewById(view, i);
                        if (editText3 != null) {
                            i = R.id.forgot_edit_repwd;
                            EditText editText4 = (EditText) ViewBindings.findChildViewById(view, i);
                            if (editText4 != null) {
                                i = R.id.forgot_txt_code;
                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView != null) {
                                    i = R.id.forgot_txt_mail;
                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView2 != null) {
                                        i = R.id.forgot_txt_pwd;
                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView3 != null) {
                                            i = R.id.forgot_txt_repwd;
                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView4 != null) {
                                                i = R.id.title_txt_name;
                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView5 != null) {
                                                    return new ActivityForgotBinding((LinearLayout) view, button, button2, editText, editText2, editText3, editText4, textView, textView2, textView3, textView4, textView5);
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

    public static ActivityForgotBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityForgotBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_forgot, viewGroup, false);
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
