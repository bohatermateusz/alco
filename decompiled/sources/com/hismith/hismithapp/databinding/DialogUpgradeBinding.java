package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogUpgradeBinding implements ViewBinding {
    public final LinearLayout dialogUpgradeLayoutContent;
    public final TextView dialogUpgradeTxtCancel;
    public final TextView dialogUpgradeTxtDesc;
    public final TextView dialogUpgradeTxtTitle;
    public final TextView dialogUpgradeTxtUpgrade;
    public final TextView dialogUpgradeTxtVersion;
    private final LinearLayout rootView;

    private DialogUpgradeBinding(LinearLayout linearLayout, LinearLayout linearLayout2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = linearLayout;
        this.dialogUpgradeLayoutContent = linearLayout2;
        this.dialogUpgradeTxtCancel = textView;
        this.dialogUpgradeTxtDesc = textView2;
        this.dialogUpgradeTxtTitle = textView3;
        this.dialogUpgradeTxtUpgrade = textView4;
        this.dialogUpgradeTxtVersion = textView5;
    }

    public static DialogUpgradeBinding bind(View view) {
        int i = R.id.dialog_upgrade_layout_content;
        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
        if (linearLayout != null) {
            i = R.id.dialog_upgrade_txt_cancel;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.dialog_upgrade_txt_desc;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.dialog_upgrade_txt_title;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        i = R.id.dialog_upgrade_txt_upgrade;
                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView4 != null) {
                            i = R.id.dialog_upgrade_txt_version;
                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView5 != null) {
                                return new DialogUpgradeBinding((LinearLayout) view, linearLayout, textView, textView2, textView3, textView4, textView5);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogUpgradeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogUpgradeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_upgrade, viewGroup, false);
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
