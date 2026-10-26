package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemBluetoothBinding implements ViewBinding {
    public final ImageView bgBattery;
    public final Button bluetoothItemBtnConn;
    public final AppCompatTextView bluetoothItemTxtName;
    public final TextView bluetoothItemTxtRename;
    public final ImageView imgBattery;
    public final ConstraintLayout layoutBattery;
    private final LinearLayout rootView;
    public final View viewBattery;

    private ItemBluetoothBinding(LinearLayout linearLayout, ImageView imageView, Button button, AppCompatTextView appCompatTextView, TextView textView, ImageView imageView2, ConstraintLayout constraintLayout, View view) {
        this.rootView = linearLayout;
        this.bgBattery = imageView;
        this.bluetoothItemBtnConn = button;
        this.bluetoothItemTxtName = appCompatTextView;
        this.bluetoothItemTxtRename = textView;
        this.imgBattery = imageView2;
        this.layoutBattery = constraintLayout;
        this.viewBattery = view;
    }

    public static ItemBluetoothBinding bind(View view) {
        View findChildViewById;
        int i = R.id.bg_battery;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.bluetooth_item_btn_conn;
            Button button = (Button) ViewBindings.findChildViewById(view, i);
            if (button != null) {
                i = R.id.bluetooth_item_txt_name;
                AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                if (appCompatTextView != null) {
                    i = R.id.bluetooth_item_txt_rename;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.img_battery;
                        ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView2 != null) {
                            i = R.id.layout_battery;
                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_battery))) != null) {
                                return new ItemBluetoothBinding((LinearLayout) view, imageView, button, appCompatTextView, textView, imageView2, constraintLayout, findChildViewById);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemBluetoothBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemBluetoothBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_bluetooth, viewGroup, false);
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
