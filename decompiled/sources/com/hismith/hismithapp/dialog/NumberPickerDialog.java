package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.content.res.Resources;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.WindowManager;
import android.widget.NumberPicker;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.QNumberPicker;
import java.lang.reflect.Field;

/* loaded from: classes3.dex */
public abstract class NumberPickerDialog extends Dialog implements View.OnClickListener {
    private Activity mActivity;
    private int mMilli;
    private QNumberPicker mPickerDot;
    private QNumberPicker mPickerMilli;
    private QNumberPicker mPickerSeconds;
    private int mSeconds;
    private TextView mTxtSubmit;
    private TextView mTxtTitle;

    public NumberPickerDialog(Activity activity, int i, int i2) {
        super(activity, R.style.ActionSheetDialogStyle);
        this.mActivity = activity;
        this.mSeconds = i;
        this.mMilli = i2;
    }

    private void setNumberPickerDividerColor(NumberPicker numberPicker) {
        for (Field field : NumberPicker.class.getDeclaredFields()) {
            if (field.getName().equals("mSelectionDivider")) {
                field.setAccessible(true);
                try {
                    field.set(numberPicker, new ColorDrawable(this.mActivity.getResources().getColor(R.color.app_style_color)));
                    return;
                } catch (Resources.NotFoundException e) {
                    e.printStackTrace();
                    return;
                } catch (IllegalAccessException e2) {
                    e2.printStackTrace();
                    return;
                } catch (IllegalArgumentException e3) {
                    e3.printStackTrace();
                    return;
                }
            }
        }
    }

    private void setViewLocation() {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        this.mActivity.getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int i = displayMetrics.heightPixels;
        WindowManager.LayoutParams attributes = getWindow().getAttributes();
        attributes.softInputMode = 16;
        attributes.x = 0;
        attributes.y = i;
        attributes.width = -1;
        attributes.height = -2;
        onWindowAttributesChanged(attributes);
    }

    public abstract void btnSubmit(int i, int i2);

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.dialog_txt_submit) {
            btnSubmit(this.mPickerSeconds.getValue(), this.mPickerMilli.getValue());
        }
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_number_picker);
        this.mTxtTitle = (TextView) findViewById(R.id.dialog_txt_title);
        this.mTxtSubmit = (TextView) findViewById(R.id.dialog_txt_submit);
        this.mPickerSeconds = (QNumberPicker) findViewById(R.id.dialog_picker_seconds);
        this.mPickerDot = (QNumberPicker) findViewById(R.id.dialog_picker_dot);
        this.mPickerMilli = (QNumberPicker) findViewById(R.id.dialog_picker_milli);
        LanguageUtil.setText(this.mActivity, this.mTxtTitle, "defaultSelect");
        LanguageUtil.setText(this.mActivity, this.mTxtSubmit, "submit");
        this.mPickerDot.setDisplayedValues(new String[]{"."});
        this.mPickerDot.setMaxValue(0);
        this.mPickerDot.setValue(0);
        this.mPickerSeconds.setMaxValue(99);
        this.mPickerMilli.setMaxValue(9);
        this.mPickerSeconds.setValue(this.mSeconds);
        this.mPickerMilli.setValue(this.mMilli);
        setNumberPickerDividerColor(this.mPickerSeconds);
        setNumberPickerDividerColor(this.mPickerDot);
        setNumberPickerDividerColor(this.mPickerMilli);
        this.mTxtSubmit.setOnClickListener(this);
        setViewLocation();
    }
}
