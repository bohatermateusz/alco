package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import cn.bingoogolapple.qrcode.zxing.ZXingView;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityQrcodeBinding implements ViewBinding {
    public final LinearLayout layoutTitle;
    private final FrameLayout rootView;
    public final ImageView scanImgLight;
    public final ImageView scanImgPic;
    public final TextView titleTxtName;
    public final ZXingView zxingview;

    private ActivityQrcodeBinding(FrameLayout frameLayout, LinearLayout linearLayout, ImageView imageView, ImageView imageView2, TextView textView, ZXingView zXingView) {
        this.rootView = frameLayout;
        this.layoutTitle = linearLayout;
        this.scanImgLight = imageView;
        this.scanImgPic = imageView2;
        this.titleTxtName = textView;
        this.zxingview = zXingView;
    }

    public static ActivityQrcodeBinding bind(View view) {
        int i = R.id.layout_title;
        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
        if (linearLayout != null) {
            i = R.id.scan_img_light;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.scan_img_pic;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.title_txt_name;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.zxingview;
                        ZXingView zXingView = (ZXingView) ViewBindings.findChildViewById(view, i);
                        if (zXingView != null) {
                            return new ActivityQrcodeBinding((FrameLayout) view, linearLayout, imageView, imageView2, textView, zXingView);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityQrcodeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityQrcodeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_qrcode, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public FrameLayout getRoot() {
        return this.rootView;
    }
}
