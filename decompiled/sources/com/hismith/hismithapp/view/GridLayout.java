package com.hismith.hismithapp.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.database.DataSetObserver;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.XBaseAdapter;

/* loaded from: classes3.dex */
public class GridLayout extends ViewGroup {
    protected XBaseAdapter mAdapter;
    private int mHorizontalSpace;
    private int mMaxItem;
    protected DataSetObserver mObserver;
    private int mSpan;
    private int mVerticalSpace;

    public GridLayout(Context context) {
        this(context, null);
    }

    public GridLayout(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public GridLayout(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mSpan = 3;
        this.mHorizontalSpace = 0;
        this.mVerticalSpace = 0;
        this.mMaxItem = 9;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, R.styleable.GridLayout);
        this.mSpan = obtainStyledAttributes.getInteger(R.styleable.GridLayout_gridSpan, this.mSpan);
        this.mHorizontalSpace = (int) obtainStyledAttributes.getDimension(R.styleable.GridLayout_gridHorizontalSpace, this.mHorizontalSpace);
        this.mVerticalSpace = (int) obtainStyledAttributes.getDimension(R.styleable.GridLayout_gridVerticalSpace, this.mVerticalSpace);
        this.mMaxItem = obtainStyledAttributes.getInteger(R.styleable.GridLayout_gridMaxItem, this.mMaxItem);
        obtainStyledAttributes.recycle();
    }

    @Override // android.view.ViewGroup, android.view.View
    protected void onLayout(boolean z, int i, int i2, int i3, int i4) {
        int min = Math.min(getChildCount(), this.mMaxItem);
        if (min <= 0) {
            return;
        }
        int paddingLeft = getPaddingLeft();
        int i5 = 0;
        for (int i6 = 0; i6 < min; i6++) {
            View childAt = getChildAt(i6);
            if (childAt.getVisibility() != 8) {
                int measuredWidth = childAt.getMeasuredWidth();
                int measuredHeight = childAt.getMeasuredHeight();
                childAt.layout(paddingLeft, i5, paddingLeft + measuredWidth, i5 + measuredHeight);
                paddingLeft += measuredWidth + this.mHorizontalSpace;
                if ((i6 + 1) % this.mSpan == 0) {
                    paddingLeft = getPaddingLeft();
                    i5 += measuredHeight + this.mVerticalSpace;
                }
            }
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        int size = View.MeasureSpec.getSize(i);
        int paddingLeft = (size - getPaddingLeft()) - getPaddingRight();
        int i3 = this.mHorizontalSpace;
        int i4 = this.mSpan;
        int i5 = (paddingLeft - (i3 * (i4 - 1))) / i4;
        int min = Math.min(getChildCount(), this.mMaxItem);
        if (min <= 0) {
            setMeasuredDimension(0, 0);
            return;
        }
        for (int i6 = 0; i6 < min; i6++) {
            View childAt = getChildAt(i6);
            int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(i5, 1073741824);
            measureChild(childAt, makeMeasureSpec, makeMeasureSpec);
        }
        int i7 = this.mSpan;
        setMeasuredDimension(size, (i5 * (min % i7 == 0 ? min / i7 : (min / i7) + 1)) + (this.mVerticalSpace * ((min - 1) / i7)));
    }

    protected void resetLayout() {
        removeAllViews();
        int min = Math.min(this.mAdapter.getCount(), this.mMaxItem);
        for (int i = 0; i < min; i++) {
            addView(this.mAdapter.getView(i, this));
        }
    }

    public void setAdapter(XBaseAdapter xBaseAdapter) {
        DataSetObserver dataSetObserver;
        XBaseAdapter xBaseAdapter2 = this.mAdapter;
        if (xBaseAdapter2 != null && (dataSetObserver = this.mObserver) != null) {
            xBaseAdapter2.unregisterDataSetObserver(dataSetObserver);
            this.mAdapter = null;
            this.mObserver = null;
        }
        if (xBaseAdapter == null) {
            throw new NullPointerException("FlowBaseAdapter is null");
        }
        this.mAdapter = xBaseAdapter;
        resetLayout();
        DataSetObserver dataSetObserver2 = new DataSetObserver() { // from class: com.hismith.hismithapp.view.GridLayout.1
            @Override // android.database.DataSetObserver
            public void onChanged() {
                GridLayout.this.resetLayout();
            }
        };
        this.mObserver = dataSetObserver2;
        this.mAdapter.registerDataSetObserver(dataSetObserver2);
    }
}
