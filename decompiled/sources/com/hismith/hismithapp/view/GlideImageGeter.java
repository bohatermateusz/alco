package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.text.Html;
import android.widget.TextView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.load.resource.gif.GifDrawable;
import com.bumptech.glide.request.BaseRequestOptions;
import com.bumptech.glide.request.RequestOptions;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.target.Target;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.glide.GlideRoundTransform;
import java.util.HashSet;
import java.util.Iterator;
import jp.wasabeef.glide.transformations.BlurTransformation;

/* loaded from: classes3.dex */
public class GlideImageGeter implements Html.ImageGetter {
    private int blurRadius;
    private int dp;
    private HashSet<GifDrawable> gifDrawables;
    private final Context mContext;
    private final TextView mTextView;
    private HashSet<Target> targets;

    private class BitmapTarget extends SimpleTarget<Bitmap> {
        private final UrlDrawable urlDrawable;

        public BitmapTarget(UrlDrawable urlDrawable) {
            this.urlDrawable = urlDrawable;
        }

        public void onResourceReady(Bitmap bitmap, Transition<? super Bitmap> transition) {
            BitmapDrawable bitmapDrawable = new BitmapDrawable(GlideImageGeter.this.mContext.getResources(), bitmap);
            int measuredWidth = GlideImageGeter.this.mTextView.getMeasuredWidth();
            Rect rect = new Rect(0, 20, measuredWidth, (bitmapDrawable.getIntrinsicHeight() * measuredWidth) / bitmapDrawable.getIntrinsicWidth());
            bitmapDrawable.setBounds(rect);
            this.urlDrawable.setBounds(rect);
            this.urlDrawable.setDrawable(bitmapDrawable);
            GlideImageGeter.this.mTextView.setText(GlideImageGeter.this.mTextView.getText());
            GlideImageGeter.this.mTextView.invalidate();
        }

        @Override // com.bumptech.glide.request.target.Target
        public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
            onResourceReady((Bitmap) obj, (Transition<? super Bitmap>) transition);
        }
    }

    private class GifTarget extends SimpleTarget<GifDrawable> {
        private final UrlDrawable urlDrawable;

        private GifTarget(UrlDrawable urlDrawable) {
            this.urlDrawable = urlDrawable;
        }

        public void onResourceReady(GifDrawable gifDrawable, Transition<? super GifDrawable> transition) {
            int measuredWidth = GlideImageGeter.this.mTextView.getMeasuredWidth();
            Rect rect = new Rect(0, 20, measuredWidth, (gifDrawable.getIntrinsicHeight() * measuredWidth) / gifDrawable.getIntrinsicWidth());
            gifDrawable.setBounds(rect);
            this.urlDrawable.setBounds(rect);
            this.urlDrawable.setDrawable(gifDrawable);
            GlideImageGeter.this.gifDrawables.add(gifDrawable);
            gifDrawable.setCallback(GlideImageGeter.this.mTextView);
            gifDrawable.start();
            gifDrawable.setLoopCount(-1);
            GlideImageGeter.this.mTextView.setText(GlideImageGeter.this.mTextView.getText());
            GlideImageGeter.this.mTextView.invalidate();
        }

        @Override // com.bumptech.glide.request.target.Target
        public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
            onResourceReady((GifDrawable) obj, (Transition<? super GifDrawable>) transition);
        }
    }

    public GlideImageGeter(Context context, TextView textView) {
        this(context, textView, 0);
    }

    public GlideImageGeter(Context context, TextView textView, int i) {
        this.dp = 0;
        this.blurRadius = 0;
        this.mContext = context;
        this.mTextView = textView;
        this.targets = new HashSet<>();
        this.gifDrawables = new HashSet<>();
        this.dp = i;
    }

    public GlideImageGeter(Context context, TextView textView, int i, int i2) {
        this.dp = 0;
        this.blurRadius = 0;
        this.mContext = context;
        this.mTextView = textView;
        this.targets = new HashSet<>();
        this.gifDrawables = new HashSet<>();
        this.dp = i;
        this.blurRadius = i2;
    }

    private static boolean isGif(String str) {
        int lastIndexOf = str.lastIndexOf(46);
        return lastIndexOf > 0 && "gif".toUpperCase().equals(str.substring(lastIndexOf + 1).toUpperCase());
    }

    @Override // android.text.Html.ImageGetter
    public Drawable getDrawable(String str) {
        RequestBuilder<Bitmap> load;
        Target bitmapTarget;
        UrlDrawable urlDrawable = new UrlDrawable();
        RequestOptions requestOptions = new RequestOptions();
        if (this.blurRadius > 0) {
            requestOptions.transform(new BlurTransformation(this.blurRadius), new GlideRoundTransform(this.mContext, this.dp));
        } else {
            requestOptions.transform(new GlideRoundTransform(this.mContext, this.dp));
        }
        boolean isGif = isGif(str);
        Context context = this.mContext;
        if (isGif) {
            load = Glide.with(context).asGif().apply((BaseRequestOptions<?>) requestOptions).load(str);
            bitmapTarget = new GifTarget(urlDrawable);
        } else {
            load = Glide.with(context).asBitmap().apply((BaseRequestOptions<?>) requestOptions).load(str);
            bitmapTarget = new BitmapTarget(urlDrawable);
        }
        this.targets.add(bitmapTarget);
        load.into((RequestBuilder<Bitmap>) bitmapTarget);
        return urlDrawable;
    }

    public void recycle() {
        this.targets.clear();
        Iterator<GifDrawable> it = this.gifDrawables.iterator();
        while (it.hasNext()) {
            GifDrawable next = it.next();
            next.setCallback(null);
            next.recycle();
        }
        this.gifDrawables.clear();
    }
}
