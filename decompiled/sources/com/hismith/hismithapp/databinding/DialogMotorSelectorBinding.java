package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Group;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CenterSeekBar;
import com.hismith.hismithapp.view.NumberRangeEditText;
import com.hismith.hismithapp.view.QNumberPicker;

/* loaded from: classes3.dex */
public final class DialogMotorSelectorBinding implements ViewBinding {
    public final Button dialogConfirmBtnCancel;
    public final Button dialogConfirmBtnConfirm;
    public final ConstraintLayout dialogConfirmLayoutContent;
    public final LinearLayout dialogConfirmLayoutTitle;
    public final TextView dialogConfirmTxtTitle;
    public final NumberRangeEditText editDelay;
    public final NumberRangeEditText editOnTime;
    public final NumberRangeEditText editOnValue;
    public final NumberRangeEditText editThreshold;
    public final Group groupCondition;
    public final Group groupDelay;
    public final Group groupFactor;
    public final Group groupSwitchValue;
    public final ImageView imgMainLogo;
    public final ImageView imgMotor1;
    public final ImageView imgMotor1Select;
    public final ImageView imgMotor2;
    public final ImageView imgMotor2Select;
    public final ImageView imgMotorNone;
    public final ImageView imgMotorNoneSelect;
    public final QNumberPicker pickerCondition;
    private final ConstraintLayout rootView;
    public final CenterSeekBar seekMotorScale;
    public final TextView txtDelay;
    public final TextView txtMotorScale;
    public final TextView txtOnTimeUnit;
    public final TextView txtSwitchOn;
    public final TextView txtSwitchTime;

    private DialogMotorSelectorBinding(ConstraintLayout constraintLayout, Button button, Button button2, ConstraintLayout constraintLayout2, LinearLayout linearLayout, TextView textView, NumberRangeEditText numberRangeEditText, NumberRangeEditText numberRangeEditText2, NumberRangeEditText numberRangeEditText3, NumberRangeEditText numberRangeEditText4, Group group, Group group2, Group group3, Group group4, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, QNumberPicker qNumberPicker, CenterSeekBar centerSeekBar, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6) {
        this.rootView = constraintLayout;
        this.dialogConfirmBtnCancel = button;
        this.dialogConfirmBtnConfirm = button2;
        this.dialogConfirmLayoutContent = constraintLayout2;
        this.dialogConfirmLayoutTitle = linearLayout;
        this.dialogConfirmTxtTitle = textView;
        this.editDelay = numberRangeEditText;
        this.editOnTime = numberRangeEditText2;
        this.editOnValue = numberRangeEditText3;
        this.editThreshold = numberRangeEditText4;
        this.groupCondition = group;
        this.groupDelay = group2;
        this.groupFactor = group3;
        this.groupSwitchValue = group4;
        this.imgMainLogo = imageView;
        this.imgMotor1 = imageView2;
        this.imgMotor1Select = imageView3;
        this.imgMotor2 = imageView4;
        this.imgMotor2Select = imageView5;
        this.imgMotorNone = imageView6;
        this.imgMotorNoneSelect = imageView7;
        this.pickerCondition = qNumberPicker;
        this.seekMotorScale = centerSeekBar;
        this.txtDelay = textView2;
        this.txtMotorScale = textView3;
        this.txtOnTimeUnit = textView4;
        this.txtSwitchOn = textView5;
        this.txtSwitchTime = textView6;
    }

    public static DialogMotorSelectorBinding bind(View view) {
        int i = R.id.dialog_confirm_btn_cancel;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.dialog_confirm_btn_confirm;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.dialog_confirm_layout_content;
                ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                if (constraintLayout != null) {
                    i = R.id.dialog_confirm_layout_title;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.dialog_confirm_txt_title;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.edit_delay;
                            NumberRangeEditText numberRangeEditText = (NumberRangeEditText) ViewBindings.findChildViewById(view, i);
                            if (numberRangeEditText != null) {
                                i = R.id.edit_on_time;
                                NumberRangeEditText numberRangeEditText2 = (NumberRangeEditText) ViewBindings.findChildViewById(view, i);
                                if (numberRangeEditText2 != null) {
                                    i = R.id.edit_on_value;
                                    NumberRangeEditText numberRangeEditText3 = (NumberRangeEditText) ViewBindings.findChildViewById(view, i);
                                    if (numberRangeEditText3 != null) {
                                        i = R.id.edit_threshold;
                                        NumberRangeEditText numberRangeEditText4 = (NumberRangeEditText) ViewBindings.findChildViewById(view, i);
                                        if (numberRangeEditText4 != null) {
                                            i = R.id.group_condition;
                                            Group group = (Group) ViewBindings.findChildViewById(view, i);
                                            if (group != null) {
                                                i = R.id.group_delay;
                                                Group group2 = (Group) ViewBindings.findChildViewById(view, i);
                                                if (group2 != null) {
                                                    i = R.id.group_factor;
                                                    Group group3 = (Group) ViewBindings.findChildViewById(view, i);
                                                    if (group3 != null) {
                                                        i = R.id.group_switch_value;
                                                        Group group4 = (Group) ViewBindings.findChildViewById(view, i);
                                                        if (group4 != null) {
                                                            i = R.id.img_main_logo;
                                                            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView != null) {
                                                                i = R.id.img_motor1;
                                                                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                if (imageView2 != null) {
                                                                    i = R.id.img_motor1_select;
                                                                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                    if (imageView3 != null) {
                                                                        i = R.id.img_motor2;
                                                                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                        if (imageView4 != null) {
                                                                            i = R.id.img_motor2_select;
                                                                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                            if (imageView5 != null) {
                                                                                i = R.id.img_motor_none;
                                                                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                if (imageView6 != null) {
                                                                                    i = R.id.img_motor_none_select;
                                                                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                    if (imageView7 != null) {
                                                                                        i = R.id.picker_condition;
                                                                                        QNumberPicker qNumberPicker = (QNumberPicker) ViewBindings.findChildViewById(view, i);
                                                                                        if (qNumberPicker != null) {
                                                                                            i = R.id.seek_motor_scale;
                                                                                            CenterSeekBar centerSeekBar = (CenterSeekBar) ViewBindings.findChildViewById(view, i);
                                                                                            if (centerSeekBar != null) {
                                                                                                i = R.id.txt_delay;
                                                                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView2 != null) {
                                                                                                    i = R.id.txt_motor_scale;
                                                                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView3 != null) {
                                                                                                        i = R.id.txt_on_time_unit;
                                                                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView4 != null) {
                                                                                                            i = R.id.txt_switch_on;
                                                                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (textView5 != null) {
                                                                                                                i = R.id.txt_switch_time;
                                                                                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView6 != null) {
                                                                                                                    return new DialogMotorSelectorBinding((ConstraintLayout) view, button, button2, constraintLayout, linearLayout, textView, numberRangeEditText, numberRangeEditText2, numberRangeEditText3, numberRangeEditText4, group, group2, group3, group4, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, qNumberPicker, centerSeekBar, textView2, textView3, textView4, textView5, textView6);
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

    public static DialogMotorSelectorBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogMotorSelectorBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_motor_selector, viewGroup, false);
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
