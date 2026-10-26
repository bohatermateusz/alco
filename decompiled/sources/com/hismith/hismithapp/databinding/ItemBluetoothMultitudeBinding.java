package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CenterSeekBar;

/* loaded from: classes3.dex */
public final class ItemBluetoothMultitudeBinding implements ViewBinding {
    public final ImageView bgBattery;
    public final Button btnDeviceConn;
    public final ImageView imgBattery;
    public final ImageView imgDevice;
    public final ImageView imgMotor1;
    public final ImageView imgMotor1Link;
    public final ImageView imgMotor1Motor1;
    public final ImageView imgMotor2;
    public final ImageView imgMotor2Link;
    public final ImageView imgMotor2Motor2;
    public final TextView itemTxtRename;
    public final ConstraintLayout layoutBattery;
    public final RadioButton radioMain;
    private final LinearLayout rootView;
    public final CenterSeekBar seekMotor1Scale;
    public final CenterSeekBar seekMotor2Scale;
    public final TextView txtDeviceName;
    public final TextView txtMotor1Scale;
    public final TextView txtMotor2Scale;
    public final View viewBattery;

    private ItemBluetoothMultitudeBinding(LinearLayout linearLayout, ImageView imageView, Button button, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, TextView textView, ConstraintLayout constraintLayout, RadioButton radioButton, CenterSeekBar centerSeekBar, CenterSeekBar centerSeekBar2, TextView textView2, TextView textView3, TextView textView4, View view) {
        this.rootView = linearLayout;
        this.bgBattery = imageView;
        this.btnDeviceConn = button;
        this.imgBattery = imageView2;
        this.imgDevice = imageView3;
        this.imgMotor1 = imageView4;
        this.imgMotor1Link = imageView5;
        this.imgMotor1Motor1 = imageView6;
        this.imgMotor2 = imageView7;
        this.imgMotor2Link = imageView8;
        this.imgMotor2Motor2 = imageView9;
        this.itemTxtRename = textView;
        this.layoutBattery = constraintLayout;
        this.radioMain = radioButton;
        this.seekMotor1Scale = centerSeekBar;
        this.seekMotor2Scale = centerSeekBar2;
        this.txtDeviceName = textView2;
        this.txtMotor1Scale = textView3;
        this.txtMotor2Scale = textView4;
        this.viewBattery = view;
    }

    public static ItemBluetoothMultitudeBinding bind(View view) {
        View findChildViewById;
        int i = R.id.bg_battery;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.btn_device_conn;
            Button button = (Button) ViewBindings.findChildViewById(view, i);
            if (button != null) {
                i = R.id.img_battery;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.img_device;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.img_motor1;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.img_motor1_link;
                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView5 != null) {
                                i = R.id.img_motor1_motor1;
                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView6 != null) {
                                    i = R.id.img_motor2;
                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView7 != null) {
                                        i = R.id.img_motor2_link;
                                        ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView8 != null) {
                                            i = R.id.img_motor2_motor2;
                                            ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView9 != null) {
                                                i = R.id.item_txt_rename;
                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView != null) {
                                                    i = R.id.layout_battery;
                                                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                    if (constraintLayout != null) {
                                                        i = R.id.radio_main;
                                                        RadioButton radioButton = (RadioButton) ViewBindings.findChildViewById(view, i);
                                                        if (radioButton != null) {
                                                            i = R.id.seek_motor1_scale;
                                                            CenterSeekBar centerSeekBar = (CenterSeekBar) ViewBindings.findChildViewById(view, i);
                                                            if (centerSeekBar != null) {
                                                                i = R.id.seek_motor2_scale;
                                                                CenterSeekBar centerSeekBar2 = (CenterSeekBar) ViewBindings.findChildViewById(view, i);
                                                                if (centerSeekBar2 != null) {
                                                                    i = R.id.txt_device_name;
                                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView2 != null) {
                                                                        i = R.id.txt_motor1_scale;
                                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView3 != null) {
                                                                            i = R.id.txt_motor2_scale;
                                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView4 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_battery))) != null) {
                                                                                return new ItemBluetoothMultitudeBinding((LinearLayout) view, imageView, button, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, imageView8, imageView9, textView, constraintLayout, radioButton, centerSeekBar, centerSeekBar2, textView2, textView3, textView4, findChildViewById);
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
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemBluetoothMultitudeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemBluetoothMultitudeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_bluetooth_multitude, viewGroup, false);
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
