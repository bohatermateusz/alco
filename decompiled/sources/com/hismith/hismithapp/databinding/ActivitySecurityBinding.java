package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivitySecurityBinding implements ViewBinding {
    public final TitleBarBinding include;
    private final LinearLayout rootView;
    public final TextView securityTxtResetPassword;
    public final TextView securityTxtUnsubscribe;

    private ActivitySecurityBinding(LinearLayout linearLayout, TitleBarBinding titleBarBinding, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.include = titleBarBinding;
        this.securityTxtResetPassword = textView;
        this.securityTxtUnsubscribe = textView2;
    }

    public static ActivitySecurityBinding bind(View view) {
        int i = R.id.include;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById != null) {
            TitleBarBinding bind = TitleBarBinding.bind(findChildViewById);
            int i2 = R.id.security_txt_resetPassword;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i2);
            if (textView != null) {
                i2 = R.id.security_txt_unsubscribe;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i2);
                if (textView2 != null) {
                    return new ActivitySecurityBinding((LinearLayout) view, bind, textView, textView2);
                }
            }
            i = i2;
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivitySecurityBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivitySecurityBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_security, viewGroup, false);
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
