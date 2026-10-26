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
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class FragmentDeviceUiDoubleBinding implements ViewBinding {
    public final ImageView imgDeviceFun1;
    public final ImageView imgDeviceFun2;
    public final CircleImageView imgDeviceLogo;
    public final TextView iplayTxtSpeed;
    public final TextView iplayTxtVibrate;
    private final ConstraintLayout rootView;

    private FragmentDeviceUiDoubleBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, TextView textView, TextView textView2) {
        this.rootView = constraintLayout;
        this.imgDeviceFun1 = imageView;
        this.imgDeviceFun2 = imageView2;
        this.imgDeviceLogo = circleImageView;
        this.iplayTxtSpeed = textView;
        this.iplayTxtVibrate = textView2;
    }

    public static FragmentDeviceUiDoubleBinding bind(View view) {
        int i = R.id.img_device_fun1;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_device_fun2;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_device_logo;
                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                if (circleImageView != null) {
                    i = R.id.iplay_txt_speed;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.iplay_txt_vibrate;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            return new FragmentDeviceUiDoubleBinding((ConstraintLayout) view, imageView, imageView2, circleImageView, textView, textView2);
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentDeviceUiDoubleBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentDeviceUiDoubleBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_device_ui_double, viewGroup, false);
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
