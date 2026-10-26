package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.drawable.ColorDrawable;
import android.view.View;
import android.widget.NumberPicker;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.QNumberPicker;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BottomPopupView;
import java.lang.reflect.Field;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Locale;

/* loaded from: classes3.dex */
public abstract class DatePickerDialog extends BottomPopupView implements View.OnClickListener, NumberPicker.OnValueChangeListener {
    private Calendar mCalendar;
    private Context mContext;
    private int mDay;
    private Calendar mMaxCalendar;
    private int mMonth;
    private QNumberPicker mPickerDay;
    private QNumberPicker mPickerMonth;
    private QNumberPicker mPickerYear;
    private TextView mTxtSubmit;
    private TextView mTxtTitle;
    private int mYear;

    public DatePickerDialog(Context context, int i, int i2, int i3) {
        super(context);
        this.mContext = context;
        this.mYear = i;
        this.mMonth = i2;
        this.mDay = i3;
        Calendar calendar = Calendar.getInstance();
        this.mCalendar = calendar;
        calendar.set(1, this.mYear);
        this.mCalendar.set(2, this.mMonth);
        this.mCalendar.set(5, this.mDay);
    }

    private void setNumberPickerDividerColor(NumberPicker numberPicker) {
        for (Field field : NumberPicker.class.getDeclaredFields()) {
            if (field.getName().equals("mSelectionDivider")) {
                field.setAccessible(true);
                try {
                    field.set(numberPicker, new ColorDrawable(this.mContext.getResources().getColor(R.color.app_style_color)));
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

    public abstract void btnSubmit(int i, int i2, int i3);

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_date_picker;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.dialog_picker_txt_submit) {
            btnSubmit(this.mPickerYear.getValue(), this.mPickerMonth.getValue(), this.mPickerDay.getValue());
        }
        dismiss();
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        this.mTxtTitle = (TextView) findViewById(R.id.dialog_txt_title);
        this.mTxtSubmit = (TextView) findViewById(R.id.dialog_picker_txt_submit);
        this.mPickerYear = (QNumberPicker) findViewById(R.id.dialog_picker_year);
        this.mPickerMonth = (QNumberPicker) findViewById(R.id.dialog_picker_month);
        this.mPickerDay = (QNumberPicker) findViewById(R.id.dialog_picker_day);
        LanguageUtil.setText(this.mContext, this.mTxtTitle, "defaultSelect");
        LanguageUtil.setText(this.mContext, this.mTxtSubmit, "submit");
        Calendar calendar = Calendar.getInstance();
        this.mMaxCalendar = calendar;
        calendar.add(1, -18);
        this.mPickerYear.setMinValue(1900);
        this.mPickerYear.setMaxValue(this.mMaxCalendar.get(1));
        int i = this.mYear;
        if (i < 1900) {
            this.mYear = 1900;
        } else if (i > this.mMaxCalendar.get(1)) {
            this.mYear = this.mMaxCalendar.get(1);
        }
        this.mPickerYear.setValue(this.mYear);
        this.mPickerYear.setWrapSelectorWheel(false);
        this.mPickerYear.setDescendantFocusability(393216);
        this.mPickerMonth.setMinValue(1);
        int value = this.mPickerYear.getValue();
        int i2 = this.mMaxCalendar.get(1);
        QNumberPicker qNumberPicker = this.mPickerMonth;
        if (value == i2) {
            qNumberPicker.setMaxValue(this.mMaxCalendar.get(2) + 1);
        } else {
            qNumberPicker.setMaxValue(12);
        }
        this.mPickerMonth.setValue(this.mMonth + 1);
        this.mPickerMonth.setWrapSelectorWheel(false);
        this.mPickerMonth.setDescendantFocusability(393216);
        this.mPickerDay.setMinValue(1);
        int actualMaximum = this.mMaxCalendar.getActualMaximum(5);
        if (this.mPickerYear.getValue() == this.mMaxCalendar.get(1) && this.mPickerMonth.getValue() == this.mMaxCalendar.get(2) + 1) {
            actualMaximum = this.mMaxCalendar.get(5);
        }
        this.mPickerDay.setMaxValue(actualMaximum);
        this.mPickerDay.setValue(this.mDay);
        this.mPickerDay.setWrapSelectorWheel(false);
        this.mPickerDay.setDescendantFocusability(393216);
        setNumberPickerDividerColor(this.mPickerYear);
        setNumberPickerDividerColor(this.mPickerMonth);
        setNumberPickerDividerColor(this.mPickerDay);
        this.mPickerYear.setOnValueChangedListener(this);
        this.mPickerMonth.setOnValueChangedListener(this);
        this.mTxtSubmit.setOnClickListener(this);
    }

    @Override // android.widget.NumberPicker.OnValueChangeListener
    public void onValueChange(NumberPicker numberPicker, int i, int i2) {
        String format = String.format(Locale.CHINA, "%d-%d", Integer.valueOf(this.mPickerYear.getValue()), Integer.valueOf(this.mPickerMonth.getValue()));
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM", Locale.CHINA);
        Calendar calendar = Calendar.getInstance();
        try {
            calendar.setTime(simpleDateFormat.parse(format));
        } catch (ParseException e) {
            e.printStackTrace();
        }
        int value = this.mPickerYear.getValue();
        int i3 = this.mMaxCalendar.get(1);
        QNumberPicker qNumberPicker = this.mPickerMonth;
        if (value == i3) {
            qNumberPicker.setMaxValue(this.mMaxCalendar.get(2) + 1);
            QNumberPicker qNumberPicker2 = this.mPickerMonth;
            qNumberPicker2.setValue(Math.min(qNumberPicker2.getValue(), this.mMaxCalendar.get(2) + 1));
        } else {
            qNumberPicker.setMaxValue(12);
        }
        int value2 = this.mPickerDay.getValue();
        int actualMaximum = calendar.getActualMaximum(5);
        if (this.mPickerYear.getValue() == this.mMaxCalendar.get(1) && this.mPickerMonth.getValue() == this.mMaxCalendar.get(2) + 1) {
            actualMaximum = this.mMaxCalendar.get(5);
        }
        this.mPickerDay.setMaxValue(actualMaximum);
        this.mPickerDay.setValue(Math.min(value2, actualMaximum));
    }

    public void showDialog() {
        new XPopup.Builder(this.mContext).autoFocusEditText(false).autoOpenSoftInput(false).asCustom(this).show();
    }
}
