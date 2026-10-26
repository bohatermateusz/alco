package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Paint;
import android.text.TextPaint;
import android.util.AttributeSet;
import androidx.appcompat.widget.AppCompatTextView;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class FitHeightTextView extends AppCompatTextView {
    private float mMaxTextSize;
    private float mMinTextSize;
    private Paint mTextPaint;

    public FitHeightTextView(Context context) {
        super(context);
        this.mMinTextSize = 8.0f;
    }

    public FitHeightTextView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mMinTextSize = 8.0f;
        setGravity(getGravity() | 16);
        setLines(1);
        initialise();
    }

    private void initialise() {
        TextPaint textPaint = new TextPaint();
        this.mTextPaint = textPaint;
        textPaint.set((Paint) getPaint());
        this.mMaxTextSize = getTextSize();
    }

    public static float px2sp(Context context, float f) {
        return f / context.getResources().getDisplayMetrics().scaledDensity;
    }

    private void refitText(String str, int i) {
        if (i > 0) {
            int paddingTop = (i - getPaddingTop()) - getPaddingBottom();
            float f = this.mMaxTextSize;
            LogUtil.e("最初字体大小：" + f);
            this.mTextPaint.setTextSize(f);
            LogUtil.e("字体高度：" + (this.mTextPaint.descent() - this.mTextPaint.ascent()) + ", 控件高度：" + paddingTop);
            float f2 = paddingTop;
            if (this.mTextPaint.descent() - this.mTextPaint.ascent() > f2) {
                LogUtil.e("大于高度，开始变小");
                while (true) {
                    f -= 1.0f;
                    float f3 = this.mMinTextSize;
                    if (f <= f3) {
                        f = f3;
                        break;
                    }
                    LogUtil.e("设置字体大小重新判断：" + f + ", 字体高度：" + (this.mTextPaint.descent() - this.mTextPaint.ascent()));
                    this.mTextPaint.setTextSize(f);
                    if (this.mTextPaint.descent() - this.mTextPaint.ascent() <= f2) {
                        break;
                    }
                }
            } else if (this.mTextPaint.descent() - this.mTextPaint.ascent() < f2) {
                LogUtil.e("小于高度，开始变大");
                do {
                    f += 1.0f;
                    LogUtil.e("设置字体大小重新判断：" + f + ", 字体高度：" + (this.mTextPaint.descent() - this.mTextPaint.ascent()));
                    this.mTextPaint.setTextSize(f);
                } while (this.mTextPaint.descent() - this.mTextPaint.ascent() < f2);
            }
            LogUtil.e("最终字体大小：" + f + ", 字体高度：" + (this.mTextPaint.descent() - this.mTextPaint.ascent()));
            setTextSize(px2sp(getContext(), f));
        }
    }

    @Override // android.view.View
    protected void onSizeChanged(int i, int i2, int i3, int i4) {
        if (i2 != i4) {
            refitText(getText().toString(), i2);
        }
    }

    @Override // androidx.appcompat.widget.AppCompatTextView, android.widget.TextView
    protected void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        refitText(charSequence.toString(), getHeight());
        super.onTextChanged(charSequence, i, i2, i3);
    }
}
