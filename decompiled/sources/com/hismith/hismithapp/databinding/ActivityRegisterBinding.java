package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatCheckBox;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityRegisterBinding implements ViewBinding {
    public final Button registerBtnGetcode;
    public final Button registerBtnRegister;
    public final AppCompatCheckBox registerCheckTerms;
    public final EditText registerEditCode;
    public final EditText registerEditMail;
    public final EditText registerEditPwd;
    public final EditText registerEditRepwd;
    public final TextView registerTxtCode;
    public final TextView registerTxtMail;
    public final TextView registerTxtPwd;
    public final TextView registerTxtRepwd;
    public final TextView registerTxtTerms;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityRegisterBinding(LinearLayout linearLayout, Button button, Button button2, AppCompatCheckBox appCompatCheckBox, EditText editText, EditText editText2, EditText editText3, EditText editText4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6) {
        this.rootView = linearLayout;
        this.registerBtnGetcode = button;
        this.registerBtnRegister = button2;
        this.registerCheckTerms = appCompatCheckBox;
        this.registerEditCode = editText;
        this.registerEditMail = editText2;
        this.registerEditPwd = editText3;
        this.registerEditRepwd = editText4;
        this.registerTxtCode = textView;
        this.registerTxtMail = textView2;
        this.registerTxtPwd = textView3;
        this.registerTxtRepwd = textView4;
        this.registerTxtTerms = textView5;
        this.titleTxtName = textView6;
    }

    public static ActivityRegisterBinding bind(View view) {
        int i = R.id.register_btn_getcode;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.register_btn_register;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.register_check_terms;
                AppCompatCheckBox appCompatCheckBox = (AppCompatCheckBox) ViewBindings.findChildViewById(view, i);
                if (appCompatCheckBox != null) {
                    i = R.id.register_edit_code;
                    EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText != null) {
                        i = R.id.register_edit_mail;
                        EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                        if (editText2 != null) {
                            i = R.id.register_edit_pwd;
                            EditText editText3 = (EditText) ViewBindings.findChildViewById(view, i);
                            if (editText3 != null) {
                                i = R.id.register_edit_repwd;
                                EditText editText4 = (EditText) ViewBindings.findChildViewById(view, i);
                                if (editText4 != null) {
                                    i = R.id.register_txt_code;
                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView != null) {
                                        i = R.id.register_txt_mail;
                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView2 != null) {
                                            i = R.id.register_txt_pwd;
                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView3 != null) {
                                                i = R.id.register_txt_repwd;
                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView4 != null) {
                                                    i = R.id.register_txt_terms;
                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView5 != null) {
                                                        i = R.id.title_txt_name;
                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView6 != null) {
                                                            return new ActivityRegisterBinding((LinearLayout) view, button, button2, appCompatCheckBox, editText, editText2, editText3, editText4, textView, textView2, textView3, textView4, textView5, textView6);
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
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityRegisterBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityRegisterBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_register, viewGroup, false);
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
