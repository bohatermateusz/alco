package com.hismith.hismithapp.dialog;

import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.NumberPicker;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.core.motion.utils.TypedValues;
import androidx.constraintlayout.widget.Group;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.device_rule.ConditionType;
import com.hismith.hismithapp.device_rule.SpeedChannel;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.view.CenterSeekBar;
import com.hismith.hismithapp.view.NumberRangeEditText;
import java.math.BigDecimal;
import java.math.RoundingMode;

/* loaded from: classes3.dex */
public class SelectMotorDialog extends Dialog {
    private final String address;
    private final String cancelText;
    private final String confirmText;
    private final ControlRuleManager.ControlRule controlRule;
    private NumberRangeEditText editDelay;
    private NumberRangeEditText editOnTime;
    private NumberRangeEditText editOnValue;
    private NumberRangeEditText editThreshold;
    private final SpeedChannel followerChannel;
    private ImageView imgMotor1;
    private ImageView imgMotor1Select;
    private ImageView imgMotor2;
    private ImageView imgMotor2Select;
    private ImageView imgMotorNone;
    private ImageView imgMotorNoneSelect;
    private OnRuleChangedListener listener;
    private NumberPicker pickerCondition;
    private final int position;
    private CenterSeekBar seekMotorScale;
    private final String title;
    private TextView txtMotorScale;

    public interface OnRuleChangedListener {
        void onConfirm(int i, String str, ControlRuleManager.ControlRule controlRule);
    }

    public SelectMotorDialog(Context context, int i, String str, ControlRuleManager.ControlRule controlRule, SpeedChannel speedChannel, String str2, String str3, String str4) {
        super(context, R.style.MyDialog);
        this.position = i;
        this.address = str;
        this.controlRule = controlRule;
        this.followerChannel = speedChannel;
        this.title = str2;
        this.cancelText = str3;
        this.confirmText = str4;
    }

    private void adjustDialogSize() {
        Window window = getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(getContext()).x * 0.9d);
            window.setAttributes(attributes);
        }
    }

    private void initData() {
        SpeedChannel speedChannel = this.followerChannel;
        SpeedChannel speedChannel2 = SpeedChannel.CHANNEL_A;
        ControlRuleManager.ControlRule controlRule = this.controlRule;
        SpeedChannel speedChannel3 = speedChannel == speedChannel2 ? controlRule.getSpeedChannel() : controlRule.getVibrateChannel();
        SpeedChannel speedChannel4 = SpeedChannel.CHANNEL_NONE;
        ImageView imageView = this.imgMotorNoneSelect;
        if (speedChannel3 == speedChannel4) {
            imageView.setVisibility(0);
        } else {
            imageView.setVisibility(4);
        }
        boolean hasFun = ProductUtil.hasFun(0);
        ImageView imageView2 = this.imgMotor1;
        if (hasFun) {
            imageView2.setVisibility(0);
            Glide.with(getContext()).load(ProductUtil.getProductFun(0).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.imgMotor1);
            this.imgMotor1Select.setVisibility(speedChannel3 == SpeedChannel.CHANNEL_A ? 0 : 4);
        } else {
            imageView2.setVisibility(8);
        }
        boolean hasFun2 = ProductUtil.hasFun(1);
        ImageView imageView3 = this.imgMotor2;
        if (hasFun2) {
            imageView3.setVisibility(0);
            Glide.with(getContext()).load(ProductUtil.getProductFun(1).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.imgMotor2);
            this.imgMotor2Select.setVisibility(speedChannel3 == SpeedChannel.CHANNEL_B ? 0 : 4);
        } else {
            imageView3.setVisibility(8);
        }
        if (this.seekMotorScale.getVisibility() == 0) {
            SpeedChannel speedChannel5 = this.followerChannel;
            SpeedChannel speedChannel6 = SpeedChannel.CHANNEL_A;
            ControlRuleManager.ControlRule controlRule2 = this.controlRule;
            float factorA = speedChannel5 == speedChannel6 ? controlRule2.getFactorA() : controlRule2.getFactorB();
            this.txtMotorScale.setText(String.valueOf(factorA));
            this.seekMotorScale.setProgress((int) ((factorA * 10.0f) + 20.0f));
        }
        this.pickerCondition.setValue(this.controlRule.getCondition().ordinal());
        this.editThreshold.setNumberValue(this.controlRule.getThreshold());
        if (this.followerChannel == SpeedChannel.CHANNEL_A) {
            this.editOnValue.setNumberValue(this.controlRule.getOnValueA());
            this.editOnTime.setNumberValue(this.controlRule.getDurationA());
        } else if (this.followerChannel == SpeedChannel.CHANNEL_B) {
            this.editOnValue.setNumberValue(this.controlRule.getOnValueB());
            this.editOnTime.setNumberValue(this.controlRule.getDurationB());
        }
        this.editDelay.setNumberValue(this.controlRule.getSwitchDelay());
    }

    private void initEvent() {
        this.imgMotorNone.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.SelectMotorDialog$$ExternalSyntheticLambda0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                SelectMotorDialog.this.m588xca21136a(view);
            }
        });
        this.imgMotor1.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.SelectMotorDialog$$ExternalSyntheticLambda1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                SelectMotorDialog.this.m589xc9aaad6b(view);
            }
        });
        this.imgMotor2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.SelectMotorDialog$$ExternalSyntheticLambda2
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                SelectMotorDialog.this.m590xc934476c(view);
            }
        });
        if (this.seekMotorScale.getVisibility() == 0) {
            this.seekMotorScale.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.dialog.SelectMotorDialog.1
                @Override // android.widget.SeekBar.OnSeekBarChangeListener
                public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                    float floatValue = new BigDecimal(String.valueOf((i - 20) / 10.0f)).setScale(1, RoundingMode.HALF_UP).floatValue();
                    SpeedChannel speedChannel = SelectMotorDialog.this.followerChannel;
                    SpeedChannel speedChannel2 = SpeedChannel.CHANNEL_A;
                    SelectMotorDialog selectMotorDialog = SelectMotorDialog.this;
                    if (speedChannel == speedChannel2) {
                        selectMotorDialog.controlRule.setFactorA(floatValue);
                    } else {
                        selectMotorDialog.controlRule.setFactorB(floatValue);
                    }
                    SelectMotorDialog.this.txtMotorScale.setText(String.valueOf(floatValue));
                }

                @Override // android.widget.SeekBar.OnSeekBarChangeListener
                public void onStartTrackingTouch(SeekBar seekBar) {
                }

                @Override // android.widget.SeekBar.OnSeekBarChangeListener
                public void onStopTrackingTouch(SeekBar seekBar) {
                }
            });
        }
        this.pickerCondition.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() { // from class: com.hismith.hismithapp.dialog.SelectMotorDialog$$ExternalSyntheticLambda3
            @Override // android.widget.NumberPicker.OnValueChangeListener
            public final void onValueChange(NumberPicker numberPicker, int i, int i2) {
                SelectMotorDialog.this.m591xc8bde16d(numberPicker, i, i2);
            }
        });
    }

    private void initView() {
        View inflate = LayoutInflater.from(getContext()).inflate(R.layout.dialog_motor_selector, (ViewGroup) null);
        setContentView(inflate);
        TextView textView = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_title);
        ImageView imageView = (ImageView) inflate.findViewById(R.id.img_main_logo);
        this.imgMotorNoneSelect = (ImageView) inflate.findViewById(R.id.img_motor_none_select);
        this.imgMotor1Select = (ImageView) inflate.findViewById(R.id.img_motor1_select);
        this.imgMotor2Select = (ImageView) inflate.findViewById(R.id.img_motor2_select);
        this.imgMotorNone = (ImageView) inflate.findViewById(R.id.img_motor_none);
        this.imgMotor1 = (ImageView) inflate.findViewById(R.id.img_motor1);
        this.imgMotor2 = (ImageView) inflate.findViewById(R.id.img_motor2);
        this.seekMotorScale = (CenterSeekBar) inflate.findViewById(R.id.seek_motor_scale);
        this.txtMotorScale = (TextView) inflate.findViewById(R.id.txt_motor_scale);
        this.pickerCondition = (NumberPicker) inflate.findViewById(R.id.picker_condition);
        this.editThreshold = (NumberRangeEditText) inflate.findViewById(R.id.edit_threshold);
        TextView textView2 = (TextView) inflate.findViewById(R.id.txt_switch_on);
        this.editOnValue = (NumberRangeEditText) inflate.findViewById(R.id.edit_on_value);
        TextView textView3 = (TextView) inflate.findViewById(R.id.txt_switch_time);
        TextView textView4 = (TextView) inflate.findViewById(R.id.txt_on_time_unit);
        this.editOnTime = (NumberRangeEditText) inflate.findViewById(R.id.edit_on_time);
        TextView textView5 = (TextView) inflate.findViewById(R.id.txt_delay);
        this.editDelay = (NumberRangeEditText) inflate.findViewById(R.id.edit_delay);
        Group group = (Group) inflate.findViewById(R.id.group_factor);
        Group group2 = (Group) inflate.findViewById(R.id.group_condition);
        Group group3 = (Group) inflate.findViewById(R.id.group_switch_value);
        Group group4 = (Group) inflate.findViewById(R.id.group_delay);
        Glide.with(getContext()).load(MyApplication.getApplication().getmProductModel().getLogo()).placeholder(R.mipmap.ic_app_logo).into(imageView);
        Glide.with(getContext()).load(Integer.valueOf(R.mipmap.ic_mode_select)).placeholder(R.mipmap.ic_mode_select).into(this.imgMotorNoneSelect);
        Glide.with(getContext()).load(Integer.valueOf(R.mipmap.ic_mode_select)).placeholder(R.mipmap.ic_mode_select).into(this.imgMotor1Select);
        Glide.with(getContext()).load(Integer.valueOf(R.mipmap.ic_mode_select)).placeholder(R.mipmap.ic_mode_select).into(this.imgMotor2Select);
        this.imgMotorNoneSelect.setVisibility(4);
        this.imgMotor1Select.setVisibility(4);
        this.imgMotor2Select.setVisibility(4);
        this.imgMotorNone.setImageResource(R.mipmap.ic_iplay_close_mode);
        boolean equals = Constans.DEVICE_TYPE_FUN.equals(MyApplication.getApplication().getmProductModel().getDeviceType());
        CenterSeekBar centerSeekBar = this.seekMotorScale;
        if (equals) {
            centerSeekBar.setVisibility(8);
            this.txtMotorScale.setVisibility(8);
        } else {
            centerSeekBar.setVisibility(0);
            this.txtMotorScale.setVisibility(0);
        }
        String[] strArr = new String[ConditionType.values().length];
        for (int i = 0; i < ConditionType.values().length; i++) {
            strArr[i] = ConditionType.values()[i].getDisplayName();
        }
        this.pickerCondition.setMinValue(0);
        this.pickerCondition.setMaxValue(ConditionType.values().length - 1);
        this.pickerCondition.setDisplayedValues(strArr);
        this.pickerCondition.setWrapSelectorWheel(false);
        this.pickerCondition.setDescendantFocusability(393216);
        LanguageUtil.setText(getContext(), textView2, "linked_speed");
        textView3.setText(LanguageUtil.getLanguageByKey(getContext(), TypedValues.TransitionType.S_DURATION));
        LanguageUtil.setText(getContext(), textView4, "second_unit");
        this.editOnTime.setRange(0, 100);
        this.editDelay.setRange(0, 100);
        LanguageUtil.setText(getContext(), textView5, "delay_unit");
        if (ProductUtil.isFType()) {
            if (ProductUtil.isFType(getContext(), this.address)) {
                group.setVisibility(8);
                group2.setVisibility(8);
                group3.setVisibility(8);
                group4.setVisibility(0);
            } else {
                group.setVisibility(8);
                group2.setVisibility(8);
                group3.setVisibility(0);
                group4.setVisibility(8);
            }
        } else if (ProductUtil.isFType(getContext(), this.address)) {
            group.setVisibility(8);
            group2.setVisibility(0);
            group3.setVisibility(8);
            group4.setVisibility(8);
        } else {
            group.setVisibility(0);
            group2.setVisibility(8);
            group3.setVisibility(8);
            group4.setVisibility(8);
        }
        Button button = (Button) inflate.findViewById(R.id.dialog_confirm_btn_cancel);
        Button button2 = (Button) inflate.findViewById(R.id.dialog_confirm_btn_confirm);
        LanguageUtil.setText(getContext(), button, "cancel");
        LanguageUtil.setText(getContext(), button2, "confirm");
        if (this.title != null) {
            LanguageUtil.setText(getContext(), textView, this.title);
        }
        if (this.cancelText != null) {
            LanguageUtil.setText(getContext(), button, this.cancelText);
        }
        if (this.confirmText != null) {
            LanguageUtil.setText(getContext(), button2, this.confirmText);
        }
        button.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.SelectMotorDialog$$ExternalSyntheticLambda4
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                SelectMotorDialog.this.m592x798c116d(view);
            }
        });
        button2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.SelectMotorDialog$$ExternalSyntheticLambda5
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                SelectMotorDialog.this.m593x7915ab6e(view);
            }
        });
    }

    private void onConfirmClicked() {
        this.controlRule.setThreshold(this.editThreshold.getNumberValue());
        if (this.followerChannel == SpeedChannel.CHANNEL_A) {
            this.controlRule.setOnValueA(this.editOnValue.getNumberValue());
            this.controlRule.setDurationA(this.editOnTime.getNumberValue());
        } else if (this.followerChannel == SpeedChannel.CHANNEL_B) {
            this.controlRule.setOnValueB(this.editOnValue.getNumberValue());
            this.controlRule.setDurationB(this.editOnTime.getNumberValue());
        }
        this.controlRule.setSwitchDelay(this.editDelay.getNumberValue());
        ControlRuleManager.getInstance().setDeviceRule(this.address, this.controlRule);
        ControlRuleManager.getInstance().saveDeviceRulesForLeaderId();
        OnRuleChangedListener onRuleChangedListener = this.listener;
        if (onRuleChangedListener != null) {
            onRuleChangedListener.onConfirm(this.position, this.address, this.controlRule);
        }
        dismiss();
    }

    /* renamed from: lambda$initEvent$2$com-hismith-hismithapp-dialog-SelectMotorDialog, reason: not valid java name */
    /* synthetic */ void m588xca21136a(View view) {
        SpeedChannel speedChannel = this.followerChannel;
        SpeedChannel speedChannel2 = SpeedChannel.CHANNEL_A;
        ControlRuleManager.ControlRule controlRule = this.controlRule;
        if (speedChannel == speedChannel2) {
            controlRule.setSpeedChannel(SpeedChannel.CHANNEL_NONE);
        } else {
            controlRule.setVibrateChannel(SpeedChannel.CHANNEL_NONE);
        }
        this.imgMotorNoneSelect.setVisibility(0);
        this.imgMotor1Select.setVisibility(4);
        this.imgMotor2Select.setVisibility(4);
    }

    /* renamed from: lambda$initEvent$3$com-hismith-hismithapp-dialog-SelectMotorDialog, reason: not valid java name */
    /* synthetic */ void m589xc9aaad6b(View view) {
        SpeedChannel speedChannel = this.followerChannel;
        SpeedChannel speedChannel2 = SpeedChannel.CHANNEL_A;
        ControlRuleManager.ControlRule controlRule = this.controlRule;
        if (speedChannel == speedChannel2) {
            controlRule.setSpeedChannel(SpeedChannel.CHANNEL_A);
        } else {
            controlRule.setVibrateChannel(SpeedChannel.CHANNEL_A);
        }
        this.imgMotorNoneSelect.setVisibility(4);
        this.imgMotor1Select.setVisibility(0);
        this.imgMotor2Select.setVisibility(4);
    }

    /* renamed from: lambda$initEvent$4$com-hismith-hismithapp-dialog-SelectMotorDialog, reason: not valid java name */
    /* synthetic */ void m590xc934476c(View view) {
        SpeedChannel speedChannel = this.followerChannel;
        SpeedChannel speedChannel2 = SpeedChannel.CHANNEL_A;
        ControlRuleManager.ControlRule controlRule = this.controlRule;
        if (speedChannel == speedChannel2) {
            controlRule.setSpeedChannel(SpeedChannel.CHANNEL_B);
        } else {
            controlRule.setVibrateChannel(SpeedChannel.CHANNEL_B);
        }
        this.imgMotorNoneSelect.setVisibility(4);
        this.imgMotor1Select.setVisibility(4);
        this.imgMotor2Select.setVisibility(0);
    }

    /* renamed from: lambda$initEvent$5$com-hismith-hismithapp-dialog-SelectMotorDialog, reason: not valid java name */
    /* synthetic */ void m591xc8bde16d(NumberPicker numberPicker, int i, int i2) {
        this.controlRule.setCondition(ConditionType.values()[i2]);
    }

    /* renamed from: lambda$initView$0$com-hismith-hismithapp-dialog-SelectMotorDialog, reason: not valid java name */
    /* synthetic */ void m592x798c116d(View view) {
        dismiss();
    }

    /* renamed from: lambda$initView$1$com-hismith-hismithapp-dialog-SelectMotorDialog, reason: not valid java name */
    /* synthetic */ void m593x7915ab6e(View view) {
        onConfirmClicked();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        initView();
        initData();
        initEvent();
        adjustDialogSize();
        setCanceledOnTouchOutside(true);
    }

    public void setOnRuleChangedListener(OnRuleChangedListener onRuleChangedListener) {
        this.listener = onRuleChangedListener;
    }
}
