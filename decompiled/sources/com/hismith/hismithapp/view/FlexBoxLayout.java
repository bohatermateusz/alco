package com.hismith.hismithapp.view;

import android.content.Context;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;

/* loaded from: classes3.dex */
public class FlexBoxLayout extends ViewGroup {
    public FlexBoxLayout(Context context) {
        this(context, null);
    }

    public FlexBoxLayout(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public FlexBoxLayout(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(AttributeSet attributeSet) {
        return new ViewGroup.MarginLayoutParams(getContext(), attributeSet);
    }

    @Override // android.view.ViewGroup, android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        int paddingTop = getPaddingTop();
        int i5 = 0;
        int i6 = 0;
        for (int i7 = 0; i7 < getChildCount(); i7++) {
            View childAt = getChildAt(i7);
            if (childAt.getVisibility() == 0) {
                ViewGroup.MarginLayoutParams marginLayoutParams = (ViewGroup.MarginLayoutParams) childAt.getLayoutParams();
                int measuredWidth = childAt.getMeasuredWidth();
                int measuredHeight = childAt.getMeasuredHeight();
                if (i5 + measuredWidth + marginLayoutParams.leftMargin + marginLayoutParams.rightMargin > (getWidth() - getPaddingStart()) - getPaddingEnd()) {
                    paddingTop += i6;
                    i5 = 0;
                    i6 = 0;
                }
                int paddingStart = marginLayoutParams.leftMargin + i5 + getPaddingStart();
                int i8 = marginLayoutParams.topMargin + paddingTop;
                i5 += measuredWidth + marginLayoutParams.leftMargin + marginLayoutParams.rightMargin;
                i6 = Math.max(i6, measuredHeight + marginLayoutParams.topMargin + marginLayoutParams.bottomMargin);
                childAt.layout(paddingStart, i8, paddingStart + measuredWidth, i8 + measuredHeight);
            }
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        measureChildren(i, i2);
        int size = View.MeasureSpec.getSize(i);
        int size2 = View.MeasureSpec.getSize(i2);
        int mode = View.MeasureSpec.getMode(i);
        int mode2 = View.MeasureSpec.getMode(i2);
        int paddingLeft = getPaddingLeft() + getPaddingRight();
        int paddingTop = getPaddingTop() + getPaddingBottom();
        int i3 = 0;
        int i4 = 0;
        for (int i5 = 0; i5 < getChildCount(); i5++) {
            View childAt = getChildAt(i5);
            if (childAt.getVisibility() == 0) {
                ViewGroup.MarginLayoutParams marginLayoutParams = (ViewGroup.MarginLayoutParams) childAt.getLayoutParams();
                int measuredWidth = childAt.getMeasuredWidth() + marginLayoutParams.leftMargin + marginLayoutParams.rightMargin;
                int measuredHeight = childAt.getMeasuredHeight() + marginLayoutParams.bottomMargin + marginLayoutParams.topMargin;
                int i6 = i3 + measuredWidth;
                if (i6 <= (size - getPaddingStart()) - getPaddingEnd()) {
                    measuredHeight = Math.max(i4, measuredHeight);
                    measuredWidth = i6;
                } else {
                    paddingLeft = Math.max(paddingLeft, i3);
                    paddingTop += i4;
                }
                if (i5 == getChildCount() - 1) {
                    paddingLeft = Math.max(paddingLeft, measuredWidth);
                    paddingTop += measuredHeight;
                }
                i4 = measuredHeight;
                i3 = measuredWidth;
            }
        }
        if (mode != 1073741824) {
            size = paddingLeft;
        }
        if (mode2 != 1073741824) {
            size2 = paddingTop;
        }
        setMeasuredDimension(size, size2);
    }
}
