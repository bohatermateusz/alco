package com.hismith.hismithapp.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.NumberPicker;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class QNumberPicker extends NumberPicker {
    public QNumberPicker(Context context) {
        super(context);
    }

    public QNumberPicker(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
    }

    public QNumberPicker(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
    }

    @Override // android.view.ViewGroup
    public void addView(View view) {
        updateView(view);
        super.addView(view);
    }

    @Override // android.view.ViewGroup
    public void addView(View view, int i, ViewGroup.LayoutParams layoutParams) {
        updateView(view);
        super.addView(view, i, layoutParams);
    }

    @Override // android.view.ViewGroup, android.view.ViewManager
    public void addView(View view, ViewGroup.LayoutParams layoutParams) {
        updateView(view);
        super.addView(view, layoutParams);
    }

    public void updateView(View view) {
        if (view instanceof EditText) {
            EditText editText = (EditText) view;
            editText.setTextColor(getResources().getColor(R.color.number_picker_text_color));
            editText.setTextSize(24.0f);
            editText.getPaint().setFakeBoldText(true);
        }
    }
}
