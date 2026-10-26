package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.appcompat.widget.Toolbar;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class TitleViewBinding implements ViewBinding {
    private final Toolbar rootView;
    public final ImageView titleImgBack;
    public final LinearLayout titleLayout;
    public final TextView titleTxtEdit;
    public final AppCompatTextView titleTxtName;
    public final Toolbar toolbar;

    private TitleViewBinding(Toolbar toolbar, ImageView imageView, LinearLayout linearLayout, TextView textView, AppCompatTextView appCompatTextView, Toolbar toolbar2) {
        this.rootView = toolbar;
        this.titleImgBack = imageView;
        this.titleLayout = linearLayout;
        this.titleTxtEdit = textView;
        this.titleTxtName = appCompatTextView;
        this.toolbar = toolbar2;
    }

    public static TitleViewBinding bind(View view) {
        int i = R.id.title_img_back;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.title_layout;
            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
            if (linearLayout != null) {
                i = R.id.title_txt_edit;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    i = R.id.title_txt_name;
                    AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                    if (appCompatTextView != null) {
                        Toolbar toolbar = (Toolbar) view;
                        return new TitleViewBinding(toolbar, imageView, linearLayout, textView, appCompatTextView, toolbar);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static TitleViewBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static TitleViewBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.title_view, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public Toolbar getRoot() {
        return this.rootView;
    }
}
