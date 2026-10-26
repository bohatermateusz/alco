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
public final class ActivityAboutBinding implements ViewBinding {
    public final TextView aboutTxtAboutApp;
    public final TextView aboutTxtAboutusPrivacyNotice;
    public final TextView aboutTxtAboutusRateme;
    public final TextView aboutTxtAboutusUserTerm;
    public final TextView aboutTxtCopyright;
    public final TextView aboutTxtName;
    public final TextView aboutTxtVersion;
    public final TitleBarBinding include;
    private final LinearLayout rootView;

    private ActivityAboutBinding(LinearLayout linearLayout, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TitleBarBinding titleBarBinding) {
        this.rootView = linearLayout;
        this.aboutTxtAboutApp = textView;
        this.aboutTxtAboutusPrivacyNotice = textView2;
        this.aboutTxtAboutusRateme = textView3;
        this.aboutTxtAboutusUserTerm = textView4;
        this.aboutTxtCopyright = textView5;
        this.aboutTxtName = textView6;
        this.aboutTxtVersion = textView7;
        this.include = titleBarBinding;
    }

    public static ActivityAboutBinding bind(View view) {
        View findChildViewById;
        int i = R.id.about_txt_aboutApp;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.about_txt_aboutusPrivacyNotice;
            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView2 != null) {
                i = R.id.about_txt_aboutusRateme;
                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView3 != null) {
                    i = R.id.about_txt_aboutusUserTerm;
                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView4 != null) {
                        i = R.id.about_txt_copyright;
                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView5 != null) {
                            i = R.id.about_txt_name;
                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView6 != null) {
                                i = R.id.about_txt_version;
                                TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView7 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.include))) != null) {
                                    return new ActivityAboutBinding((LinearLayout) view, textView, textView2, textView3, textView4, textView5, textView6, textView7, TitleBarBinding.bind(findChildViewById));
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityAboutBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityAboutBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_about, viewGroup, false);
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
