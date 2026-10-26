package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/* loaded from: classes3.dex */
public class WheelView extends ScrollView {
    public static final int OFF_SET_DEFAULT = 1;
    private static final int SCROLL_DIRECTION_DOWN = 1;
    private static final int SCROLL_DIRECTION_UP = 0;
    public static final String TAG = "WheelView";
    private Context context;
    int displayItemCount;
    int initialY;
    int itemHeight;
    List<String> items;
    int newCheck;
    int offset;
    private OnWheelViewListener onWheelViewListener;
    Paint paint;
    private int scrollDirection;
    Runnable scrollerTask;
    int[] selectedAreaBorder;
    int selectedIndex;
    int viewWidth;
    private LinearLayout views;

    public static class OnWheelViewListener {
        public void onSelected(int i, String str) {
        }
    }

    public WheelView(Context context) {
        super(context);
        this.offset = 1;
        this.selectedIndex = 1;
        this.newCheck = 50;
        this.itemHeight = 0;
        this.scrollDirection = -1;
        init(context);
    }

    public WheelView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.offset = 1;
        this.selectedIndex = 1;
        this.newCheck = 50;
        this.itemHeight = 0;
        this.scrollDirection = -1;
        init(context);
    }

    public WheelView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.offset = 1;
        this.selectedIndex = 1;
        this.newCheck = 50;
        this.itemHeight = 0;
        this.scrollDirection = -1;
        init(context);
    }

    private TextView createView(String str) {
        TextView textView = new TextView(this.context);
        textView.setLayoutParams(new FrameLayout.LayoutParams(-1, -2));
        textView.setSingleLine(true);
        textView.setTextSize(2, 20.0f);
        textView.setText(str);
        textView.setGravity(17);
        int dip2px = dip2px(15.0f);
        textView.setPadding(dip2px, dip2px, dip2px, dip2px);
        if (this.itemHeight == 0) {
            this.itemHeight = getViewMeasuredHeight(textView);
            this.views.setLayoutParams(new FrameLayout.LayoutParams(-2, this.itemHeight * this.displayItemCount, 1));
            this.views.setGravity(17);
            setLayoutParams(new LinearLayout.LayoutParams(((LinearLayout.LayoutParams) getLayoutParams()).width, this.itemHeight * this.displayItemCount));
        }
        return textView;
    }

    private int dip2px(float f) {
        return (int) ((f * this.context.getResources().getDisplayMetrics().density) + 0.5f);
    }

    private int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, getResources().getDisplayMetrics());
    }

    private List<String> getItems() {
        return this.items;
    }

    private int getViewMeasuredHeight(View view) {
        view.measure(View.MeasureSpec.makeMeasureSpec(0, 0), View.MeasureSpec.makeMeasureSpec(536870911, Integer.MIN_VALUE));
        return view.getMeasuredHeight();
    }

    private void init(Context context) {
        this.context = context;
        setVerticalScrollBarEnabled(false);
        LinearLayout linearLayout = new LinearLayout(context);
        this.views = linearLayout;
        linearLayout.setOrientation(1);
        addView(this.views);
        this.scrollerTask = new Runnable() { // from class: com.hismith.hismithapp.view.WheelView.1
            @Override // java.lang.Runnable
            public void run() {
                int scrollY = WheelView.this.initialY - WheelView.this.getScrollY();
                WheelView wheelView = WheelView.this;
                if (scrollY != 0) {
                    wheelView.initialY = wheelView.getScrollY();
                    WheelView wheelView2 = WheelView.this;
                    wheelView2.postDelayed(wheelView2.scrollerTask, WheelView.this.newCheck);
                    return;
                }
                final int i = wheelView.initialY % WheelView.this.itemHeight;
                final int i2 = WheelView.this.initialY / WheelView.this.itemHeight;
                WheelView wheelView3 = WheelView.this;
                if (i == 0) {
                    wheelView3.selectedIndex = i2 + wheelView3.offset;
                    WheelView.this.onSeletedCallBack();
                    return;
                }
                int i3 = wheelView3.itemHeight / 2;
                WheelView wheelView4 = WheelView.this;
                if (i > i3) {
                    wheelView4.post(new Runnable() { // from class: com.hismith.hismithapp.view.WheelView.1.1
                        @Override // java.lang.Runnable
                        public void run() {
                            WheelView.this.smoothScrollTo(0, (WheelView.this.initialY - i) + WheelView.this.itemHeight);
                            WheelView.this.selectedIndex = i2 + WheelView.this.offset + 1;
                            WheelView.this.onSeletedCallBack();
                        }
                    });
                } else {
                    wheelView4.post(new Runnable() { // from class: com.hismith.hismithapp.view.WheelView.1.2
                        @Override // java.lang.Runnable
                        public void run() {
                            WheelView.this.smoothScrollTo(0, WheelView.this.initialY - i);
                            WheelView.this.selectedIndex = i2 + WheelView.this.offset;
                            WheelView.this.onSeletedCallBack();
                        }
                    });
                }
            }
        };
        OnWheelViewListener onWheelViewListener = this.onWheelViewListener;
        if (onWheelViewListener != null) {
            int i = this.selectedIndex;
            onWheelViewListener.onSelected(i, this.items.get(i));
        }
    }

    private void initData() {
        this.displayItemCount = (this.offset * 2) + 1;
        this.views.removeAllViews();
        Iterator<String> it = this.items.iterator();
        while (it.hasNext()) {
            this.views.addView(createView(it.next()));
        }
        refreshItemView(0);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public int[] obtainSelectedAreaBorder() {
        int[] iArr = this.selectedAreaBorder;
        if (iArr != null) {
            return iArr;
        }
        int[] iArr2 = {r1 * r4, r1 * (r4 + 1)};
        this.selectedAreaBorder = iArr2;
        int i = this.itemHeight;
        int i2 = this.offset;
        return iArr2;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void onSeletedCallBack() {
        OnWheelViewListener onWheelViewListener = this.onWheelViewListener;
        if (onWheelViewListener != null) {
            int i = this.selectedIndex;
            onWheelViewListener.onSelected(i, this.items.get(i));
        }
    }

    private void refreshItemView(int i) {
        int i2 = this.itemHeight;
        int i3 = this.offset;
        int i4 = (i / i2) + i3;
        int i5 = i % i2;
        int i6 = i / i2;
        if (i5 == 0) {
            i4 = i6 + i3;
        } else if (i5 > i2 / 2) {
            i4 = i6 + i3 + 1;
        }
        int childCount = this.views.getChildCount();
        for (int i7 = 0; i7 < childCount; i7++) {
            TextView textView = (TextView) this.views.getChildAt(i7);
            if (textView == null) {
                return;
            }
            Context context = this.context;
            if (i4 == i7) {
                textView.setTextColor(context.getResources().getColor(R.color.black));
            } else {
                textView.setTextColor(context.getResources().getColor(R.color.gray9));
            }
        }
    }

    @Override // android.widget.ScrollView
    public void fling(int i) {
        super.fling(i / 3);
    }

    public int getOffset() {
        return this.offset;
    }

    public OnWheelViewListener getOnWheelViewListener() {
        return this.onWheelViewListener;
    }

    public int getSeletedIndex() {
        return this.selectedIndex - this.offset;
    }

    public String getSeletedItem() {
        return this.items.get(this.selectedIndex);
    }

    @Override // android.view.View
    protected void onScrollChanged(int i, int i2, int i3, int i4) {
        super.onScrollChanged(i, i2, i3, i4);
        refreshItemView(i2);
        if (i2 > i4) {
            this.scrollDirection = 1;
        } else {
            this.scrollDirection = 0;
        }
    }

    @Override // android.widget.ScrollView, android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        Log.d(TAG, "w: " + i + ", h: " + i2 + ", oldw: " + i3 + ", oldh: " + i4);
        this.viewWidth = i;
        setBackgroundDrawable(null);
    }

    @Override // android.widget.ScrollView, android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        if (motionEvent.getAction() == 1) {
            startScrollerTask();
        }
        return super.onTouchEvent(motionEvent);
    }

    @Override // android.view.View
    public void setBackgroundDrawable(Drawable drawable) {
        if (this.viewWidth == 0) {
            this.viewWidth = this.views.getWidth();
            Log.d(TAG, "viewWidth: " + this.viewWidth);
        }
        if (this.paint == null) {
            Paint paint = new Paint();
            this.paint = paint;
            paint.setColor(Color.parseColor("#83cde6"));
            this.paint.setStrokeWidth(dip2px(2.0f));
        }
        super.setBackgroundDrawable(new Drawable() { // from class: com.hismith.hismithapp.view.WheelView.2
            @Override // android.graphics.drawable.Drawable
            public void draw(Canvas canvas) {
                canvas.drawLine(WheelView.this.viewWidth / 6, WheelView.this.obtainSelectedAreaBorder()[0], (WheelView.this.viewWidth * 5) / 6, WheelView.this.obtainSelectedAreaBorder()[0], WheelView.this.paint);
                canvas.drawLine(WheelView.this.viewWidth / 6, WheelView.this.obtainSelectedAreaBorder()[1], (WheelView.this.viewWidth * 5) / 6, WheelView.this.obtainSelectedAreaBorder()[1], WheelView.this.paint);
            }

            @Override // android.graphics.drawable.Drawable
            public int getOpacity() {
                return 0;
            }

            @Override // android.graphics.drawable.Drawable
            public void setAlpha(int i) {
            }

            @Override // android.graphics.drawable.Drawable
            public void setColorFilter(ColorFilter colorFilter) {
            }
        });
    }

    public void setItems(List<String> list) {
        if (this.items == null) {
            this.items = new ArrayList();
        }
        this.items.clear();
        this.items.addAll(list);
        for (int i = 0; i < this.offset; i++) {
            this.items.add(0, "");
            this.items.add("");
        }
        initData();
    }

    public void setOffset(int i) {
        this.offset = i;
    }

    public void setOnWheelViewListener(OnWheelViewListener onWheelViewListener) {
        this.onWheelViewListener = onWheelViewListener;
        int i = this.selectedIndex;
        onWheelViewListener.onSelected(i, this.items.get(i));
    }

    public void setSeletion(final int i) {
        this.selectedIndex = this.offset + i;
        post(new Runnable() { // from class: com.hismith.hismithapp.view.WheelView.3
            @Override // java.lang.Runnable
            public void run() {
                WheelView wheelView = WheelView.this;
                wheelView.smoothScrollTo(0, i * wheelView.itemHeight);
            }
        });
    }

    public void startScrollerTask() {
        this.initialY = getScrollY();
        postDelayed(this.scrollerTask, this.newCheck);
    }
}
