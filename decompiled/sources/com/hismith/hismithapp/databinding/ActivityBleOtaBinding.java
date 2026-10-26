package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityBleOtaBinding implements ViewBinding {
    public final Button otaBtnCheck;
    public final ImageView otaImgLogo;
    public final TextView otaTxtName;
    public final TextView otaTxtNameLabel;
    public final TextView otaTxtNewVersion;
    public final TextView otaTxtNewVersionLabel;
    public final TextView otaTxtVersion;
    public final TextView otaTxtVersionLabel;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityBleOtaBinding(LinearLayout linearLayout, Button button, ImageView imageView, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7) {
        this.rootView = linearLayout;
        this.otaBtnCheck = button;
        this.otaImgLogo = imageView;
        this.otaTxtName = textView;
        this.otaTxtNameLabel = textView2;
        this.otaTxtNewVersion = textView3;
        this.otaTxtNewVersionLabel = textView4;
        this.otaTxtVersion = textView5;
        this.otaTxtVersionLabel = textView6;
        this.titleTxtName = textView7;
    }

    public static ActivityBleOtaBinding bind(View view) {
        int i = R.id.ota_btn_check;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.ota_img_logo;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.ota_txt_name;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    i = R.id.ota_txt_name_label;
                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView2 != null) {
                        i = R.id.ota_txt_new_version;
                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView3 != null) {
                            i = R.id.ota_txt_new_version_label;
                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView4 != null) {
                                i = R.id.ota_txt_version;
                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView5 != null) {
                                    i = R.id.ota_txt_version_label;
                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView6 != null) {
                                        i = R.id.title_txt_name;
                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView7 != null) {
                                            return new ActivityBleOtaBinding((LinearLayout) view, button, imageView, textView, textView2, textView3, textView4, textView5, textView6, textView7);
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

    public static ActivityBleOtaBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityBleOtaBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_ble_ota, viewGroup, false);
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
