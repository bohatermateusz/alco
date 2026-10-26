package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivitySingleSexSettingsBinding implements ViewBinding {
    public final TitleBarBinding include;
    private final LinearLayout rootView;
    public final ImageView settingImgViewCommunity;
    public final ImageView settingImgViewSex;
    public final TextView settingTxtViewCommunity;
    public final TextView settingTxtViewSex;

    private ActivitySingleSexSettingsBinding(LinearLayout linearLayout, TitleBarBinding titleBarBinding, ImageView imageView, ImageView imageView2, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.include = titleBarBinding;
        this.settingImgViewCommunity = imageView;
        this.settingImgViewSex = imageView2;
        this.settingTxtViewCommunity = textView;
        this.settingTxtViewSex = textView2;
    }

    public static ActivitySingleSexSettingsBinding bind(View view) {
        int i = R.id.include;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById != null) {
            TitleBarBinding bind = TitleBarBinding.bind(findChildViewById);
            i = R.id.setting_img_viewCommunity;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.setting_img_viewSex;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.setting_txt_viewCommunity;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.setting_txt_viewSex;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            return new ActivitySingleSexSettingsBinding((LinearLayout) view, bind, imageView, imageView2, textView, textView2);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivitySingleSexSettingsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivitySingleSexSettingsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_single_sex_settings, viewGroup, false);
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
