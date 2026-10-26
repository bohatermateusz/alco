package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.appcompat.widget.Toolbar;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class TitleView2Binding implements ViewBinding {
    private final Toolbar rootView;
    public final ImageView titleImgBack;
    public final ImageView titleImgGender;
    public final ImageView titleImgMore;
    public final LinearLayout titleLayout;
    public final AppCompatTextView titleTxtName;
    public final Toolbar toolbar;

    private TitleView2Binding(Toolbar toolbar, ImageView imageView, ImageView imageView2, ImageView imageView3, LinearLayout linearLayout, AppCompatTextView appCompatTextView, Toolbar toolbar2) {
        this.rootView = toolbar;
        this.titleImgBack = imageView;
        this.titleImgGender = imageView2;
        this.titleImgMore = imageView3;
        this.titleLayout = linearLayout;
        this.titleTxtName = appCompatTextView;
        this.toolbar = toolbar2;
    }

    public static TitleView2Binding bind(View view) {
        int i = R.id.title_img_back;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.title_img_gender;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.title_img_more;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.title_layout;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.title_txt_name;
                        AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                        if (appCompatTextView != null) {
                            Toolbar toolbar = (Toolbar) view;
                            return new TitleView2Binding(toolbar, imageView, imageView2, imageView3, linearLayout, appCompatTextView, toolbar);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static TitleView2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static TitleView2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.title_view2, viewGroup, false);
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
