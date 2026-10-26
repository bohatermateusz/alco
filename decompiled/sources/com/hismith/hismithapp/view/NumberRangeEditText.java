package com.hismith.hismithapp.view;

import android.content.Context;
import android.text.Editable;
import android.text.InputFilter;
import android.text.Spanned;
import android.text.TextWatcher;
import android.util.AttributeSet;
import android.view.View;
import androidx.appcompat.widget.AppCompatEditText;

/* loaded from: classes3.dex */
public class NumberRangeEditText extends AppCompatEditText {
    private boolean isSettingText;
    private String lastValidValue;
    private int maxValue;
    private int minValue;

    private class InputFilterMinMax implements InputFilter {
        private final int max;
        private final int min;

        public InputFilterMinMax(int i, int i2) {
            this.min = i;
            this.max = i2;
        }

        @Override // android.text.InputFilter
        public CharSequence filter(CharSequence charSequence, int i, int i2, Spanned spanned, int i3, int i4) {
            if (NumberRangeEditText.this.isSettingText) {
                return null;
            }
            try {
                String obj = spanned.toString();
                String str = obj.substring(0, i3) + ((Object) charSequence.subSequence(i, i2)) + obj.substring(i4);
                if (str.isEmpty()) {
                    return null;
                }
                if (obj.equals("0") && str.length() == 1 && charSequence.length() == 1 && i3 == 0 && i4 == 1) {
                    NumberRangeEditText.this.lastValidValue = str;
                    return null;
                }
                if (!str.matches("\\d+")) {
                    return "";
                }
                int parseInt = Integer.parseInt(str);
                int i5 = this.min;
                if (parseInt < i5) {
                    return String.valueOf(i5);
                }
                int i6 = this.max;
                if (parseInt > i6) {
                    return String.valueOf(i6);
                }
                NumberRangeEditText.this.lastValidValue = str;
                return null;
            } catch (Exception unused) {
                return "";
            }
        }
    }

    private class NumberTextWatcher implements TextWatcher {
        private NumberTextWatcher() {
        }

        @Override // android.text.TextWatcher
        public void afterTextChanged(Editable editable) {
            if (NumberRangeEditText.this.isSettingText) {
                return;
            }
            String obj = editable.toString();
            if (obj.isEmpty()) {
                NumberRangeEditText.this.isSettingText = true;
                editable.replace(0, editable.length(), "0");
                NumberRangeEditText.this.lastValidValue = "0";
                NumberRangeEditText.this.isSettingText = false;
                return;
            }
            try {
                if (!obj.matches("\\d+")) {
                    editable.replace(0, editable.length(), NumberRangeEditText.this.lastValidValue);
                    return;
                }
                if (obj.startsWith("0") && obj.length() > 1) {
                    String valueOf = String.valueOf(Integer.parseInt(obj));
                    NumberRangeEditText.this.isSettingText = true;
                    editable.replace(0, editable.length(), valueOf);
                    NumberRangeEditText.this.isSettingText = false;
                    NumberRangeEditText.this.lastValidValue = valueOf;
                    return;
                }
                int parseInt = Integer.parseInt(obj);
                int i = NumberRangeEditText.this.minValue;
                NumberRangeEditText numberRangeEditText = NumberRangeEditText.this;
                if (parseInt < i) {
                    String valueOf2 = String.valueOf(numberRangeEditText.minValue);
                    NumberRangeEditText.this.isSettingText = true;
                    editable.replace(0, editable.length(), valueOf2);
                    NumberRangeEditText.this.isSettingText = false;
                    NumberRangeEditText.this.lastValidValue = valueOf2;
                    return;
                }
                int i2 = numberRangeEditText.maxValue;
                NumberRangeEditText numberRangeEditText2 = NumberRangeEditText.this;
                if (parseInt <= i2) {
                    numberRangeEditText2.lastValidValue = obj;
                    return;
                }
                String valueOf3 = String.valueOf(numberRangeEditText2.maxValue);
                NumberRangeEditText.this.isSettingText = true;
                editable.replace(0, editable.length(), valueOf3);
                NumberRangeEditText.this.isSettingText = false;
                NumberRangeEditText.this.lastValidValue = valueOf3;
            } catch (NumberFormatException unused) {
                editable.replace(0, editable.length(), NumberRangeEditText.this.lastValidValue);
            }
        }

        @Override // android.text.TextWatcher
        public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        }

        @Override // android.text.TextWatcher
        public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        }
    }

    public NumberRangeEditText(Context context) {
        super(context);
        this.minValue = 0;
        this.maxValue = 100;
        this.lastValidValue = "0";
        this.isSettingText = false;
        init();
    }

    public NumberRangeEditText(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.minValue = 0;
        this.maxValue = 100;
        this.lastValidValue = "0";
        this.isSettingText = false;
        init();
    }

    public NumberRangeEditText(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.minValue = 0;
        this.maxValue = 100;
        this.lastValidValue = "0";
        this.isSettingText = false;
        init();
    }

    private void handleEmptyText() {
        if (this.isSettingText) {
            return;
        }
        this.isSettingText = true;
        if (getText().toString().isEmpty()) {
            setText("0");
            this.lastValidValue = "0";
        }
        this.isSettingText = false;
    }

    private void init() {
        setInputType(2);
        setText("0");
        setFilters(new InputFilter[]{new InputFilterMinMax(this.minValue, this.maxValue)});
        addTextChangedListener(new NumberTextWatcher());
        setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.view.NumberRangeEditText$$ExternalSyntheticLambda0
            @Override // android.view.View.OnFocusChangeListener
            public final void onFocusChange(View view, boolean z) {
                NumberRangeEditText.this.m632lambda$init$0$comhismithhismithappviewNumberRangeEditText(view, z);
            }
        });
    }

    public int getNumberValue() {
        try {
            String obj = getText().toString();
            if (obj.isEmpty()) {
                return 0;
            }
            return Integer.parseInt(obj);
        } catch (NumberFormatException unused) {
            return 0;
        }
    }

    /* renamed from: lambda$init$0$com-hismith-hismithapp-view-NumberRangeEditText, reason: not valid java name */
    /* synthetic */ void m632lambda$init$0$comhismithhismithappviewNumberRangeEditText(View view, boolean z) {
        if (z) {
            return;
        }
        handleEmptyText();
    }

    @Override // android.widget.TextView
    public void onEditorAction(int i) {
        super.onEditorAction(i);
        if (i == 6 || i == 5 || i == 2) {
            handleEmptyText();
        }
    }

    @Override // android.widget.TextView
    protected void onSelectionChanged(int i, int i2) {
        super.onSelectionChanged(i, i2);
        String obj = getText().toString();
        if (obj.equals("0")) {
            setSelection(obj.length());
        }
    }

    public void setNumberValue(int i) {
        int i2 = this.minValue;
        if (i < i2 || i > (i2 = this.maxValue)) {
            i = i2;
        }
        setText(String.valueOf(i));
    }

    public void setRange(int i, int i2) {
        this.minValue = i;
        this.maxValue = i2;
        setFilters(new InputFilter[]{new InputFilterMinMax(i, i2)});
    }
}
