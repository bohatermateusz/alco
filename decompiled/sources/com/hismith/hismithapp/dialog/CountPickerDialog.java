package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.app.Dialog;
import android.content.res.Resources;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.text.InputFilter;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.NumberPicker;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.QNumberPicker;
import java.lang.reflect.Field;
import java.util.List;

/* loaded from: classes3.dex */
public abstract class CountPickerDialog extends Dialog implements View.OnClickListener {
    private Activity mActivity;
    private Integer mCount;
    private int mEnd;
    private String[] mFormatArr;
    private Integer mMax;
    private Integer mMin;
    private QNumberPicker mPickerCount;
    private List<Integer> mSingleSerialList;
    private int mStart;
    private TextView mTxtSubmit;
    private TextView mTxtTitle;

    public CountPickerDialog(Activity activity, Integer num, Integer num2, Integer num3) {
        super(activity, R.style.ActionSheetDialogStyle);
        this.mStart = 0;
        this.mEnd = -1;
        this.mActivity = activity;
        this.mCount = num;
        this.mMin = num2;
        this.mMax = num3;
    }

    public CountPickerDialog(Activity activity, Integer num, Integer num2, Integer num3, String[] strArr) {
        super(activity, R.style.ActionSheetDialogStyle);
        this.mStart = 0;
        this.mEnd = -1;
        this.mActivity = activity;
        this.mCount = num;
        this.mMin = num2;
        this.mMax = num3;
        this.mFormatArr = strArr;
        if (strArr != null) {
            this.mMin = 0;
            int intValue = this.mMax.intValue();
            String[] strArr2 = this.mFormatArr;
            if (intValue > strArr2.length) {
                this.mMax = Integer.valueOf(strArr2.length);
            }
        }
    }

    private void setInputText() {
        try {
            Field declaredField = NumberPicker.class.getDeclaredField("mInputText");
            declaredField.setAccessible(true);
            ((EditText) declaredField.get(this.mPickerCount)).setFilters(new InputFilter[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
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

    public abstract void btnSubmit(int i);

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.dialog_txt_submit) {
            btnSubmit(this.mPickerCount.getValue() + this.mStart);
        }
        dismiss();
    }

    @Override // android.app.Dialog
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.dialog_count_picker);
        this.mTxtTitle = (TextView) findViewById(R.id.dialog_txt_title);
        this.mTxtSubmit = (TextView) findViewById(R.id.dialog_txt_submit);
        this.mPickerCount = (QNumberPicker) findViewById(R.id.dialog_picker_count);
        LanguageUtil.setText(this.mActivity, this.mTxtTitle, "defaultSelect");
        LanguageUtil.setText(this.mActivity, this.mTxtSubmit, "submit");
        String[] strArr = this.mFormatArr;
        if (strArr != null) {
            this.mPickerCount.setDisplayedValues(strArr);
            this.mPickerCount.setDescendantFocusability(393216);
        }
        this.mPickerCount.setMaxValue(this.mMax.intValue());
        this.mPickerCount.setMinValue(this.mMin.intValue());
        int intValue = this.mCount.intValue();
        QNumberPicker qNumberPicker = this.mPickerCount;
        if (intValue != 0) {
            qNumberPicker.setValue(this.mCount.intValue());
        } else {
            qNumberPicker.setValue(this.mMin.intValue());
        }
        setNumberPickerDividerColor(this.mPickerCount);
        this.mTxtSubmit.setOnClickListener(this);
        setViewLocation();
    }
}
