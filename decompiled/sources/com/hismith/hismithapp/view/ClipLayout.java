package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.recyclerview.widget.RecyclerView;

/* loaded from: classes3.dex */
public class ClipLayout extends FrameLayout {
    private int cachedRowHeight;
    private boolean clipFromTop;
    private int clipHeight;

    public ClipLayout(Context context) {
        super(context);
        this.clipHeight = -1;
        this.clipFromTop = false;
        this.cachedRowHeight = -1;
        init();
    }

    public ClipLayout(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.clipHeight = -1;
        this.clipFromTop = false;
        this.cachedRowHeight = -1;
        init();
    }

    public ClipLayout(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.clipHeight = -1;
        this.clipFromTop = false;
        this.cachedRowHeight = -1;
        init();
    }

    private RecyclerView findRecyclerView(ViewGroup viewGroup) {
        RecyclerView findRecyclerView;
        for (int i = 0; i < viewGroup.getChildCount(); i++) {
            View childAt = viewGroup.getChildAt(i);
            if (childAt instanceof RecyclerView) {
                return (RecyclerView) childAt;
            }
            if ((childAt instanceof ViewGroup) && (findRecyclerView = findRecyclerView((ViewGroup) childAt)) != null) {
                return findRecyclerView;
            }
        }
        return null;
    }

    private void init() {
        setWillNotDraw(false);
        setLayerType(2, null);
        setBackgroundColor(0);
    }

    private int measureRowHeight(RecyclerView recyclerView) {
        if (recyclerView.getChildCount() > 0) {
            View childAt = recyclerView.getChildAt(0);
            if (childAt.getHeight() > 0) {
                return childAt.getHeight();
            }
        }
        RecyclerView.Adapter adapter = recyclerView.getAdapter();
        RecyclerView.LayoutManager layoutManager = recyclerView.getLayoutManager();
        if (adapter != null && adapter.getItemCount() > 0 && layoutManager != null) {
            try {
                RecyclerView.ViewHolder createViewHolder = adapter.createViewHolder(recyclerView, adapter.getItemViewType(0));
                adapter.bindViewHolder(createViewHolder, 0);
                createViewHolder.itemView.measure(View.MeasureSpec.makeMeasureSpec(recyclerView.getWidth(), 1073741824), View.MeasureSpec.makeMeasureSpec(0, 0));
                int measuredHeight = createViewHolder.itemView.getMeasuredHeight();
                if (measuredHeight > 0) {
                    return measuredHeight;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return (int) TypedValue.applyDimension(1, 50.0f, getResources().getDisplayMetrics());
    }

    public void clearRowHeightCache() {
        this.cachedRowHeight = -1;
    }

    @Override // android.view.ViewGroup, android.view.View
    protected void dispatchDraw(Canvas canvas) {
        if (this.clipHeight <= 0) {
            super.dispatchDraw(canvas);
            return;
        }
        canvas.save();
        if (this.clipFromTop) {
            canvas.clipRect(0, 0, getWidth(), this.clipHeight);
        } else {
            canvas.clipRect(0, getHeight() - this.clipHeight, getWidth(), getHeight());
        }
        super.dispatchDraw(canvas);
        canvas.restore();
    }

    public int getRowHeight() {
        int i = this.cachedRowHeight;
        if (i > 0) {
            return i;
        }
        RecyclerView findRecyclerView = findRecyclerView(this);
        if (findRecyclerView == null) {
            return -1;
        }
        int measureRowHeight = measureRowHeight(findRecyclerView);
        this.cachedRowHeight = measureRowHeight;
        return measureRowHeight;
    }

    public void onRecyclerViewDataChanged() {
        clearRowHeightCache();
        getRowHeight();
    }

    public void restoreFullView() {
        setClipHeight(-1);
    }

    public void setClipFromTop(boolean z) {
        this.clipFromTop = z;
        invalidate();
    }

    public void setClipHeight(int i) {
        this.clipHeight = i;
        invalidate();
    }

    public void setDisplayLines(int i) {
        if (i <= 0) {
            setClipHeight(-1);
            return;
        }
        int rowHeight = getRowHeight();
        if (rowHeight > 0) {
            setClipHeight(rowHeight * i);
        }
    }

    public void showSingleLine() {
        int rowHeight = getRowHeight();
        if (rowHeight > 0) {
            setClipHeight(rowHeight);
        }
    }

    public void showSingleLineAndScrollToBottom() {
        final int itemCount;
        showSingleLine();
        final RecyclerView findRecyclerView = findRecyclerView(this);
        if (findRecyclerView == null || findRecyclerView.getAdapter() == null || (itemCount = findRecyclerView.getAdapter().getItemCount()) <= 0) {
            return;
        }
        findRecyclerView.post(new Runnable() { // from class: com.hismith.hismithapp.view.ClipLayout$$ExternalSyntheticLambda0
            @Override // java.lang.Runnable
            public final void run() {
                RecyclerView.this.scrollToPosition(itemCount - 1);
            }
        });
    }
}
