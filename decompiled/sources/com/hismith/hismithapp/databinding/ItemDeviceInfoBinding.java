package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemDeviceInfoBinding implements ViewBinding {
    public final ImageView imgBattery;
    public final ImageView imgDeviceFun1;
    public final ImageView imgDeviceFun2;
    public final ImageView imgDeviceLogo;
    public final ImageView iplayImgPiu;
    public final TextView iplayTxtSpeed;
    public final TextView iplayTxtVibrate;
    public final ConstraintLayout layoutBattery;
    private final ConstraintLayout rootView;
    public final View viewBattery;

    private ItemDeviceInfoBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, TextView textView, TextView textView2, ConstraintLayout constraintLayout2, View view) {
        this.rootView = constraintLayout;
        this.imgBattery = imageView;
        this.imgDeviceFun1 = imageView2;
        this.imgDeviceFun2 = imageView3;
        this.imgDeviceLogo = imageView4;
        this.iplayImgPiu = imageView5;
        this.iplayTxtSpeed = textView;
        this.iplayTxtVibrate = textView2;
        this.layoutBattery = constraintLayout2;
        this.viewBattery = view;
    }

    public static ItemDeviceInfoBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_battery;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_device_fun1;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_device_fun2;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.img_device_logo;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.iplay_img_piu;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.iplay_txt_speed;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.iplay_txt_vibrate;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    i = R.id.layout_battery;
                                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                    if (constraintLayout != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_battery))) != null) {
                                        return new ItemDeviceInfoBinding((ConstraintLayout) view, imageView, imageView2, imageView3, imageView4, imageView5, textView, textView2, constraintLayout, findChildViewById);
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

    public static ItemDeviceInfoBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemDeviceInfoBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_device_info, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public ConstraintLayout getRoot() {
        return this.rootView;
    }
}
