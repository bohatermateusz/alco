package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class DialogBleOta2Binding implements ViewBinding {
    public final ImageView dialogOtaImgBg;
    public final ImageView dialogOtaImgCancel;
    public final CircleImageView dialogOtaImgLogo;
    public final FrameLayout dialogOtaLayoutUpgrade;
    public final ProgressBar dialogOtaProgressUpgrade;
    public final TextView dialogOtaTxtError;
    public final TextView dialogOtaTxtName;
    public final TextView dialogOtaTxtNew;
    public final TextView dialogOtaTxtNewLabel;
    public final TextView dialogOtaTxtNote;
    public final TextView dialogOtaTxtOld;
    public final TextView dialogOtaTxtOldLabel;
    public final TextView dialogOtaTxtUpgrade;
    private final LinearLayout rootView;

    private DialogBleOta2Binding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, FrameLayout frameLayout, ProgressBar progressBar, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8) {
        this.rootView = linearLayout;
        this.dialogOtaImgBg = imageView;
        this.dialogOtaImgCancel = imageView2;
        this.dialogOtaImgLogo = circleImageView;
        this.dialogOtaLayoutUpgrade = frameLayout;
        this.dialogOtaProgressUpgrade = progressBar;
        this.dialogOtaTxtError = textView;
        this.dialogOtaTxtName = textView2;
        this.dialogOtaTxtNew = textView3;
        this.dialogOtaTxtNewLabel = textView4;
        this.dialogOtaTxtNote = textView5;
        this.dialogOtaTxtOld = textView6;
        this.dialogOtaTxtOldLabel = textView7;
        this.dialogOtaTxtUpgrade = textView8;
    }

    public static DialogBleOta2Binding bind(View view) {
        int i = R.id.dialog_ota_img_bg;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.dialog_ota_img_cancel;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.dialog_ota_img_logo;
                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                if (circleImageView != null) {
                    i = R.id.dialog_ota_layout_upgrade;
                    FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                    if (frameLayout != null) {
                        i = R.id.dialog_ota_progress_upgrade;
                        ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                        if (progressBar != null) {
                            i = R.id.dialog_ota_txt_error;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.dialog_ota_txt_name;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    i = R.id.dialog_ota_txt_new;
                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView3 != null) {
                                        i = R.id.dialog_ota_txt_new_label;
                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView4 != null) {
                                            i = R.id.dialog_ota_txt_note;
                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView5 != null) {
                                                i = R.id.dialog_ota_txt_old;
                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView6 != null) {
                                                    i = R.id.dialog_ota_txt_old_label;
                                                    TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView7 != null) {
                                                        i = R.id.dialog_ota_txt_upgrade;
                                                        TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView8 != null) {
                                                            return new DialogBleOta2Binding((LinearLayout) view, imageView, imageView2, circleImageView, frameLayout, progressBar, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8);
                                                        }
                                                    }
                                                }
                                            }
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

    public static DialogBleOta2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogBleOta2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_ble_ota2, viewGroup, false);
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
