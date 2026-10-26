package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import androidx.appcompat.widget.AppCompatEditText;

/* loaded from: classes.dex */
public class SearchView extends AppCompatEditText {
    private int flag;
    private String hintString;
    private int offset;
    private Drawable searchDrawable;
    private int searchHeight;
    private int searchWidth;
    private int w;

    public SearchView(Context context) {
        super(context);
        this.flag = 0;
        init();
    }

    public SearchView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.flag = 0;
        init();
    }

    public SearchView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.flag = 0;
        init();
    }

    public static int dip2px(Context context, float f) {
        return (int) ((f * context.getResources().getDisplayMetrics().density) + 0.5f);
    }

    private void getDrawable() {
        this.searchDrawable = getCompoundDrawables()[0];
    }

    private void init() {
    }

    private void setTextDrawable() {
        Drawable drawable = this.searchDrawable;
        int i = this.offset;
        drawable.setBounds(i, 0, drawable.getIntrinsicWidth() + i, this.searchDrawable.getIntrinsicHeight());
        setCompoundDrawables(this.searchDrawable, null, null, null);
    }

    @Override // android.widget.TextView, android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        this.searchWidth = getMeasuredWidth();
        this.searchHeight = getMeasuredHeight();
        this.hintString = getHint().toString();
        Paint paint = new Paint();
        Rect rect = new Rect();
        String str = this.hintString;
        paint.getTextBounds(str, 0, str.length(), rect);
        int dip2px = dip2px(getContext(), rect.width());
        this.w = dip2px;
        this.offset = ((this.searchWidth / 2) - (dip2px / 2)) - dip2px(getContext(), 26.0f);
        if (this.flag == 0) {
            setTextDrawable();
        }
        this.flag++;
    }

    @Override // android.widget.TextView
    protected void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        super.onTextChanged(charSequence, i, i2, i3);
        if (this.searchDrawable == null) {
            getDrawable();
        }
        if (length() > 0) {
            setCompoundDrawables(null, null, null, null);
        } else if (length() == 0) {
            setTextDrawable();
        }
    }
}
