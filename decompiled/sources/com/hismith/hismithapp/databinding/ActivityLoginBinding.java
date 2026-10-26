package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityLoginBinding implements ViewBinding {
    public final Button loginBtnLogin;
    public final EditText loginEditMail;
    public final EditText loginEditPwd;
    public final ImageButton loginImgbtnFacebook;
    public final ImageButton loginImgbtnGoogle;
    public final ImageButton loginImgbtnTwitter;
    public final LinearLayout loginLayout;
    public final LinearLayout loginLayoutLanguage;
    public final LinearLayout loginLayoutOffLine;
    public final TextView loginTxtForgot;
    public final TextView loginTxtLanguage;
    public final TextView loginTxtOffLine;
    public final TextView loginTxtRegister;
    public final TextView loginTxtUserTerms;
    private final LinearLayout rootView;

    private ActivityLoginBinding(LinearLayout linearLayout, Button button, EditText editText, EditText editText2, ImageButton imageButton, ImageButton imageButton2, ImageButton imageButton3, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = linearLayout;
        this.loginBtnLogin = button;
        this.loginEditMail = editText;
        this.loginEditPwd = editText2;
        this.loginImgbtnFacebook = imageButton;
        this.loginImgbtnGoogle = imageButton2;
        this.loginImgbtnTwitter = imageButton3;
        this.loginLayout = linearLayout2;
        this.loginLayoutLanguage = linearLayout3;
        this.loginLayoutOffLine = linearLayout4;
        this.loginTxtForgot = textView;
        this.loginTxtLanguage = textView2;
        this.loginTxtOffLine = textView3;
        this.loginTxtRegister = textView4;
        this.loginTxtUserTerms = textView5;
    }

    public static ActivityLoginBinding bind(View view) {
        int i = R.id.login_btn_login;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.login_edit_mail;
            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText != null) {
                i = R.id.login_edit_pwd;
                EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                if (editText2 != null) {
                    i = R.id.login_imgbtn_facebook;
                    ImageButton imageButton = (ImageButton) ViewBindings.findChildViewById(view, i);
                    if (imageButton != null) {
                        i = R.id.login_imgbtn_google;
                        ImageButton imageButton2 = (ImageButton) ViewBindings.findChildViewById(view, i);
                        if (imageButton2 != null) {
                            i = R.id.login_imgbtn_twitter;
                            ImageButton imageButton3 = (ImageButton) ViewBindings.findChildViewById(view, i);
                            if (imageButton3 != null) {
                                i = R.id.login_layout;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.login_layout_language;
                                    LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout2 != null) {
                                        i = R.id.login_layout_off_line;
                                        LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout3 != null) {
                                            i = R.id.login_txt_forgot;
                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView != null) {
                                                i = R.id.login_txt_language;
                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView2 != null) {
                                                    i = R.id.login_txt_off_line;
                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView3 != null) {
                                                        i = R.id.login_txt_register;
                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView4 != null) {
                                                            i = R.id.login_txt_userTerms;
                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView5 != null) {
                                                                return new ActivityLoginBinding((LinearLayout) view, button, editText, editText2, imageButton, imageButton2, imageButton3, linearLayout, linearLayout2, linearLayout3, textView, textView2, textView3, textView4, textView5);
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
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityLoginBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityLoginBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_login, viewGroup, false);
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
