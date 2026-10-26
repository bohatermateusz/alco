package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.TextProgressBar;

/* loaded from: classes3.dex */
public final class DialogBleOtaBinding implements ViewBinding {
    public final Button dialogOtaBtnCancel;
    public final Button dialogOtaBtnConfirm;
    public final FrameLayout dialogOtaLayoutConfirm;
    public final LinearLayout dialogOtaLayoutContent;
    public final ProgressBar dialogOtaProgressLoad;
    public final TextProgressBar dialogOtaProgressUpdate;
    public final TextView dialogOtaTxtContent;
    public final TextView dialogOtaTxtProgress;
    public final TextView dialogOtaTxtTitle;
    private final LinearLayout rootView;

    private DialogBleOtaBinding(LinearLayout linearLayout, Button button, Button button2, FrameLayout frameLayout, LinearLayout linearLayout2, ProgressBar progressBar, TextProgressBar textProgressBar, TextView textView, TextView textView2, TextView textView3) {
        this.rootView = linearLayout;
        this.dialogOtaBtnCancel = button;
        this.dialogOtaBtnConfirm = button2;
        this.dialogOtaLayoutConfirm = frameLayout;
        this.dialogOtaLayoutContent = linearLayout2;
        this.dialogOtaProgressLoad = progressBar;
        this.dialogOtaProgressUpdate = textProgressBar;
        this.dialogOtaTxtContent = textView;
        this.dialogOtaTxtProgress = textView2;
        this.dialogOtaTxtTitle = textView3;
    }

    public static DialogBleOtaBinding bind(View view) {
        int i = R.id.dialog_ota_btn_cancel;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.dialog_ota_btn_confirm;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.dialog_ota_layout_confirm;
                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout != null) {
                    i = R.id.dialog_ota_layout_content;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.dialog_ota_progress_load;
                        ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                        if (progressBar != null) {
                            i = R.id.dialog_ota_progress_update;
                            TextProgressBar textProgressBar = (TextProgressBar) ViewBindings.findChildViewById(view, i);
                            if (textProgressBar != null) {
                                i = R.id.dialog_ota_txt_content;
                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView != null) {
                                    i = R.id.dialog_ota_txt_progress;
                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView2 != null) {
                                        i = R.id.dialog_ota_txt_title;
                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView3 != null) {
                                            return new DialogBleOtaBinding((LinearLayout) view, button, button2, frameLayout, linearLayout, progressBar, textProgressBar, textView, textView2, textView3);
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

    public static DialogBleOtaBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogBleOtaBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_ble_ota, viewGroup, false);
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
