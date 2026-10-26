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
public final class FragmentDeviceUiSingleBinding implements ViewBinding {
    public final ImageView imgDeviceFun1;
    public final CircleImageView imgDeviceLogo;
    public final TextView iplayTxtSpeed;
    private final ConstraintLayout rootView;

    private FragmentDeviceUiSingleBinding(ConstraintLayout constraintLayout, ImageView imageView, CircleImageView circleImageView, TextView textView) {
        this.rootView = constraintLayout;
        this.imgDeviceFun1 = imageView;
        this.imgDeviceLogo = circleImageView;
        this.iplayTxtSpeed = textView;
    }

    public static FragmentDeviceUiSingleBinding bind(View view) {
        int i = R.id.img_device_fun1;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_device_logo;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.iplay_txt_speed;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    return new FragmentDeviceUiSingleBinding((ConstraintLayout) view, imageView, circleImageView, textView);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentDeviceUiSingleBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentDeviceUiSingleBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_device_ui_single, viewGroup, false);
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
