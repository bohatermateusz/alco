package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class PopupSpeedBinding implements ViewBinding {
    public final ImageView popupSpeedImgConfirm;
    public final SeekBar popupSpeedSeekSpeed;
    public final TextView popupSpeedTxtMax;
    public final TextView popupSpeedTxtMin;
    public final TextView popupSpeedTxtSpeed;
    public final TextView popupSpeedTxtSpeedTitle;
    private final LinearLayout rootView;

    private PopupSpeedBinding(LinearLayout linearLayout, ImageView imageView, SeekBar seekBar, TextView textView, TextView textView2, TextView textView3, TextView textView4) {
        this.rootView = linearLayout;
        this.popupSpeedImgConfirm = imageView;
        this.popupSpeedSeekSpeed = seekBar;
        this.popupSpeedTxtMax = textView;
        this.popupSpeedTxtMin = textView2;
        this.popupSpeedTxtSpeed = textView3;
        this.popupSpeedTxtSpeedTitle = textView4;
    }

    public static PopupSpeedBinding bind(View view) {
        int i = R.id.popup_speed_img_confirm;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.popup_speed_seek_speed;
            SeekBar seekBar = (SeekBar) ViewBindings.findChildViewById(view, i);
            if (seekBar != null) {
                i = R.id.popup_speed_txt_max;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    i = R.id.popup_speed_txt_min;
                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView2 != null) {
                        i = R.id.popup_speed_txt_speed;
                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView3 != null) {
                            i = R.id.popup_speed_txt_speedTitle;
                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView4 != null) {
                                return new PopupSpeedBinding((LinearLayout) view, imageView, seekBar, textView, textView2, textView3, textView4);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static PopupSpeedBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PopupSpeedBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.popup_speed, viewGroup, false);
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
