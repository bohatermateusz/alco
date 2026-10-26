package com.hismith.hismithapp.glide;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.text.SpannableStringBuilder;
import android.text.style.ImageSpan;
import android.widget.TextView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.CustomTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import java.util.ArrayList;
import java.util.List;
import jp.wasabeef.glide.transformations.RoundedCornersTransformation;

/* loaded from: classes3.dex */
public class NetworkImageLoader {

    private static class CenteredImageSpan extends ImageSpan {
        public CenteredImageSpan(Drawable drawable) {
            super(drawable);
        }

        @Override // android.text.style.DynamicDrawableSpan, android.text.style.ReplacementSpan
        public void draw(Canvas canvas, CharSequence charSequence, int i, int i2, float f, int i3, int i4, int i5, Paint paint) {
            Drawable drawable = getDrawable();
            canvas.save();
            canvas.translate(f, (i5 - drawable.getBounds().bottom) - (paint.getFontMetricsInt().descent / 2));
            drawable.draw(canvas);
            canvas.restore();
        }
    }

    private static class ImageTarget extends CustomTarget<Bitmap> {
        private final Context context;
        private final int end;
        private final int imageSizeDp;
        private final int start;
        private final TextView textView;

        public ImageTarget(Context context, TextView textView, int i, int i2, int i3) {
            this.context = context;
            this.textView = textView;
            this.start = i;
            this.end = i2;
            this.imageSizeDp = i3;
        }

        @Override // com.bumptech.glide.request.target.Target
        public void onLoadCleared(Drawable drawable) {
        }

        @Override // com.bumptech.glide.request.target.CustomTarget, com.bumptech.glide.request.target.Target
        public void onLoadFailed(Drawable drawable) {
            if (drawable != null) {
                int dpToPx = NetworkImageLoader.dpToPx(this.context, this.imageSizeDp);
                drawable.setBounds(0, 0, dpToPx, dpToPx);
                CharSequence text = this.textView.getText();
                SpannableStringBuilder spannableStringBuilder = text instanceof SpannableStringBuilder ? (SpannableStringBuilder) text : new SpannableStringBuilder(text);
                spannableStringBuilder.setSpan(new CenteredImageSpan(drawable), this.start, this.end, 33);
                this.textView.setText(spannableStringBuilder);
            }
        }

        public void onResourceReady(Bitmap bitmap, Transition<? super Bitmap> transition) {
            RoundedBitmapDrawable roundedBitmapDrawable = new RoundedBitmapDrawable(this.context, bitmap);
            int dpToPx = NetworkImageLoader.dpToPx(this.context, this.imageSizeDp);
            roundedBitmapDrawable.setBounds(0, 0, dpToPx, dpToPx);
            CenteredImageSpan centeredImageSpan = new CenteredImageSpan(roundedBitmapDrawable);
            CharSequence text = this.textView.getText();
            SpannableStringBuilder spannableStringBuilder = text instanceof SpannableStringBuilder ? (SpannableStringBuilder) text : new SpannableStringBuilder(text);
            spannableStringBuilder.setSpan(centeredImageSpan, this.start, this.end, 33);
            this.textView.setText(spannableStringBuilder);
        }

        @Override // com.bumptech.glide.request.target.Target
        public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
            onResourceReady((Bitmap) obj, (Transition<? super Bitmap>) transition);
        }
    }

    private static class RoundedBitmapDrawable extends BitmapDrawable {
        private final Bitmap bitmap;
        private final float cornerRadius;
        private final Paint paint;

        public RoundedBitmapDrawable(Context context, Bitmap bitmap) {
            super(context.getResources(), bitmap);
            this.paint = new Paint(1);
            this.bitmap = bitmap;
            this.cornerRadius = NetworkImageLoader.dpToPx(context, 10.0f);
        }

        @Override // android.graphics.drawable.BitmapDrawable, android.graphics.drawable.Drawable
        public void draw(Canvas canvas) {
            Path path = new Path();
            RectF rectF = new RectF(getBounds());
            float f = this.cornerRadius;
            path.addRoundRect(rectF, f, f, Path.Direction.CW);
            canvas.save();
            canvas.clipPath(path);
            super.draw(canvas);
            canvas.restore();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static int dpToPx(Context context, float f) {
        return (int) ((f * context.getResources().getDisplayMetrics().density) + 0.5f);
    }

    public static void loadImageIntoTextView(Context context, TextView textView, String str, String str2, int i, float f) {
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder(str);
        int length = spannableStringBuilder.length();
        spannableStringBuilder.append((CharSequence) "[image]");
        int length2 = spannableStringBuilder.length();
        textView.setText(spannableStringBuilder);
        Glide.with(context).asBitmap().load(str2).placeholder(R.mipmap.hismith_image_loading).error(R.mipmap.hismith_image_load_err).transform(new RoundedCornersTransformation(dpToPx(context, f), 0, RoundedCornersTransformation.CornerType.ALL)).into((RequestBuilder) new ImageTarget(context, textView, length, length2, i));
    }

    public static void loadImagesIntoTextView(Context context, TextView textView, String str, List<String> list, int i, float f) {
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder(str);
        ArrayList arrayList = new ArrayList();
        int i2 = 0;
        while (i2 < list.size()) {
            i2++;
            int length = spannableStringBuilder.length();
            spannableStringBuilder.append((CharSequence) ("[图片" + i2 + "]"));
            int length2 = spannableStringBuilder.length();
            arrayList.add(Integer.valueOf(length));
            arrayList.add(Integer.valueOf(length2));
        }
        textView.setText(spannableStringBuilder);
        for (int i3 = 0; i3 < list.size(); i3++) {
            int i4 = i3 * 2;
            Glide.with(context).asBitmap().load(list.get(i3)).transform(new RoundedCornersTransformation(dpToPx(context, f), 0, RoundedCornersTransformation.CornerType.ALL)).into((RequestBuilder) new ImageTarget(context, textView, ((Integer) arrayList.get(i4)).intValue(), ((Integer) arrayList.get(i4 + 1)).intValue(), i));
        }
    }
}
