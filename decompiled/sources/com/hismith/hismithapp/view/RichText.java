package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.text.Html;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.TextUtils;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.text.style.ImageSpan;
import android.util.AttributeSet;
import android.view.View;
import androidx.appcompat.widget.AppCompatTextView;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/* loaded from: classes3.dex */
public class RichText extends AppCompatTextView implements Drawable.Callback, View.OnAttachStateChangeListener {
    private GlideImageGeter glideImageGeter;
    private OnRichTextImageClickListener onRichTextImageClickListener;

    public interface OnRichTextImageClickListener {
        void imageClicked(List<String> list, int i);
    }

    public RichText(Context context) {
        this(context, null);
    }

    public RichText(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public RichText(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
    }

    private String formatUrlString(String str) {
        String[] linkString = linkString(str);
        String str2 = str.toString();
        if (linkString.length > 0) {
            for (int i = 0; i < linkString.length; i++) {
                if (!TextUtils.isEmpty(linkString[i])) {
                    str2 = str2.replace(linkString[i], "<font color='#48A7F9'> <a href=\"" + linkString[i] + "\">" + linkString[i] + "</a></font>");
                }
            }
        }
        return str2;
    }

    @Override // android.widget.TextView, android.view.View, android.graphics.drawable.Drawable.Callback
    public void invalidateDrawable(Drawable drawable) {
        invalidateOutline();
    }

    public String[] linkString(String str) {
        Matcher matcher = Pattern.compile("(?:^|[\\W])((ht|f)tp(s?):\\/\\/|www\\.)(([\\w\\-]+\\.){1,}?([\\w\\-.~]+\\/?)*[\\p{Alnum}.,%_=?&#\\-+()\\[\\]\\*$~@!:/{};']*)", 42).matcher(str);
        ArrayList arrayList = new ArrayList();
        int i = 0;
        while (matcher.find()) {
            int start = matcher.start(1);
            int end = matcher.end();
            arrayList.add(str.substring(start, end));
            LogUtil.e("matchStart:" + start + ", matchEnd:" + end + ", content:" + ((String) arrayList.get(i)));
            i++;
        }
        return (String[]) arrayList.toArray(new String[0]);
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewAttachedToWindow(View view) {
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewDetachedFromWindow(View view) {
        this.glideImageGeter.recycle();
    }

    @Override // android.view.View, android.graphics.drawable.Drawable.Callback
    public void scheduleDrawable(Drawable drawable, Runnable runnable, long j) {
    }

    public void setOnRichTextImageClickListener(OnRichTextImageClickListener onRichTextImageClickListener) {
        this.onRichTextImageClickListener = onRichTextImageClickListener;
    }

    public void setRichText(String str) {
        setRichText(str, 0);
    }

    public void setRichText(String str, int i) {
        setRichText(str, i, 0);
    }

    public void setRichText(String str, int i, int i2) {
        this.glideImageGeter = new GlideImageGeter(getContext(), this, i, i2);
        Spanned fromHtml = Html.fromHtml(str, 63, this.glideImageGeter, null);
        SpannableStringBuilder spannableStringBuilder = fromHtml instanceof SpannableStringBuilder ? (SpannableStringBuilder) fromHtml : new SpannableStringBuilder(fromHtml);
        ImageSpan[] imageSpanArr = (ImageSpan[]) spannableStringBuilder.getSpans(0, spannableStringBuilder.length(), ImageSpan.class);
        final ArrayList arrayList = new ArrayList();
        int length = imageSpanArr.length;
        for (final int i3 = 0; i3 < length; i3++) {
            ImageSpan imageSpan = imageSpanArr[i3];
            String source = imageSpan.getSource();
            int spanStart = spannableStringBuilder.getSpanStart(imageSpan);
            int spanEnd = spannableStringBuilder.getSpanEnd(imageSpan);
            arrayList.add(source);
            ClickableSpan clickableSpan = new ClickableSpan() { // from class: com.hismith.hismithapp.view.RichText.1
                @Override // android.text.style.ClickableSpan
                public void onClick(View view) {
                    if (RichText.this.onRichTextImageClickListener != null) {
                        RichText.this.onRichTextImageClickListener.imageClicked(arrayList, i3);
                    }
                }
            };
            ClickableSpan[] clickableSpanArr = (ClickableSpan[]) spannableStringBuilder.getSpans(spanStart, spanEnd, ClickableSpan.class);
            if (clickableSpanArr != null && clickableSpanArr.length != 0) {
                for (ClickableSpan clickableSpan2 : clickableSpanArr) {
                    spannableStringBuilder.removeSpan(clickableSpan2);
                }
            }
            spannableStringBuilder.setSpan(clickableSpan, spanStart, spanEnd, 33);
        }
        super.setText(fromHtml);
        LogUtil.e("spanned:" + ((Object) fromHtml));
        setMovementMethod(LinkMovementMethod.getInstance());
    }

    @Override // android.view.View, android.graphics.drawable.Drawable.Callback
    public void unscheduleDrawable(Drawable drawable, Runnable runnable) {
    }
}
