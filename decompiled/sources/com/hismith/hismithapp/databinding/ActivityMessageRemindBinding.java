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
public final class ActivityMessageRemindBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final ImageView settingImgFollowMessage;
    public final ImageView settingImgInteractiveMessage;
    public final ImageView settingImgMyMessage;
    public final ImageView settingImgProductAds;
    public final TextView settingTxtFollowMessage;
    public final TextView settingTxtInteractiveMessage;
    public final TextView settingTxtMyMessage;
    public final TextView settingTxtProductAds;
    public final TextView titleTxtName;

    private ActivityMessageRemindBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = linearLayout;
        this.settingImgFollowMessage = imageView;
        this.settingImgInteractiveMessage = imageView2;
        this.settingImgMyMessage = imageView3;
        this.settingImgProductAds = imageView4;
        this.settingTxtFollowMessage = textView;
        this.settingTxtInteractiveMessage = textView2;
        this.settingTxtMyMessage = textView3;
        this.settingTxtProductAds = textView4;
        this.titleTxtName = textView5;
    }

    public static ActivityMessageRemindBinding bind(View view) {
        int i = R.id.setting_img_followMessage;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.setting_img_interactiveMessage;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.setting_img_myMessage;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.setting_img_productAds;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.setting_txt_followMessage;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.setting_txt_interactiveMessage;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.setting_txt_myMessage;
                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView3 != null) {
                                    i = R.id.setting_txt_productAds;
                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView4 != null) {
                                        i = R.id.title_txt_name;
                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView5 != null) {
                                            return new ActivityMessageRemindBinding((LinearLayout) view, imageView, imageView2, imageView3, imageView4, textView, textView2, textView3, textView4, textView5);
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

    public static ActivityMessageRemindBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityMessageRemindBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_message_remind, viewGroup, false);
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
