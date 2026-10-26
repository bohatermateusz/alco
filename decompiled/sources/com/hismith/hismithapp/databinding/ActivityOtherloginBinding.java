package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.facebook.login.widget.LoginButton;
import com.google.android.gms.common.SignInButton;
import com.hismith.hismithapp.R;
import com.twitter.sdk.android.core.identity.TwitterLoginButton;

/* loaded from: classes3.dex */
public final class ActivityOtherloginBinding implements ViewBinding {
    public final RelativeLayout activityMain;
    public final LoginButton loginButtonFb;
    private final RelativeLayout rootView;
    public final SignInButton signInButtonGl;
    public final TwitterLoginButton twitterLoginButton;

    private ActivityOtherloginBinding(RelativeLayout relativeLayout, RelativeLayout relativeLayout2, LoginButton loginButton, SignInButton signInButton, TwitterLoginButton twitterLoginButton) {
        this.rootView = relativeLayout;
        this.activityMain = relativeLayout2;
        this.loginButtonFb = loginButton;
        this.signInButtonGl = signInButton;
        this.twitterLoginButton = twitterLoginButton;
    }

    public static ActivityOtherloginBinding bind(View view) {
        RelativeLayout relativeLayout = (RelativeLayout) view;
        int i = R.id.login_button_fb;
        LoginButton loginButton = (LoginButton) ViewBindings.findChildViewById(view, i);
        if (loginButton != null) {
            i = R.id.sign_in_button_gl;
            SignInButton signInButton = (SignInButton) ViewBindings.findChildViewById(view, i);
            if (signInButton != null) {
                i = R.id.twitter_login_button;
                TwitterLoginButton twitterLoginButton = (TwitterLoginButton) ViewBindings.findChildViewById(view, i);
                if (twitterLoginButton != null) {
                    return new ActivityOtherloginBinding(relativeLayout, relativeLayout, loginButton, signInButton, twitterLoginButton);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityOtherloginBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityOtherloginBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_otherlogin, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public RelativeLayout getRoot() {
        return this.rootView;
    }
}
