package com.hismith.hismithapp.utils;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.widget.ImageView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.BaseRequestOptions;
import com.bumptech.glide.request.RequestOptions;
import com.hismith.hismithapp.glide.GlideRoundTransform;
import jp.wasabeef.glide.transformations.BlurTransformation;

/* loaded from: classes3.dex */
public class GlideImageUtil {
    public static void loadBlurredImage(Context context, String str, ImageView imageView, int i, int i2, int i3, int i4) {
        RequestBuilder<Drawable> load = Glide.with(context).load(str);
        if (i != 0) {
            load.placeholder(i);
        }
        if (i2 != 0) {
            load.error(i2);
        }
        if (i3 == 1 && i4 == 0) {
            load.apply((BaseRequestOptions<?>) RequestOptions.bitmapTransform(new BlurTransformation(100)));
        }
        load.into(imageView);
    }

    public static void loadBlurredImage(Context context, String str, ImageView imageView, int i, int i2, int i3, int i4, int i5) {
        RequestBuilder<Drawable> load = Glide.with(context).load(str);
        RequestOptions requestOptions = new RequestOptions();
        if (i != 0) {
            load.placeholder(i);
        }
        if (i2 != 0) {
            load.error(i2);
        }
        if (i4 == 1 && i5 == 0) {
            requestOptions.transform(new BlurTransformation(100), new GlideRoundTransform(context, i3));
        } else {
            requestOptions.transform(new GlideRoundTransform(context, i3));
        }
        load.apply((BaseRequestOptions<?>) requestOptions);
        load.into(imageView);
    }
}
