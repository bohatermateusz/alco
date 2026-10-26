package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.LevelListDrawable;
import android.text.Html;
import android.widget.TextView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;

/* loaded from: classes3.dex */
public class MImageGetter implements Html.ImageGetter {
    private Context c;
    private TextView container;

    public MImageGetter(TextView textView, Context context) {
        this.c = context;
        this.container = textView;
    }

    @Override // android.text.Html.ImageGetter
    public Drawable getDrawable(String str) {
        final LevelListDrawable levelListDrawable = new LevelListDrawable();
        Glide.with(this.c).asBitmap().load(str).into((RequestBuilder<Bitmap>) new SimpleTarget<Bitmap>() { // from class: com.hismith.hismithapp.view.MImageGetter.1
            public void onResourceReady(Bitmap bitmap, Transition<? super Bitmap> transition) {
                if (bitmap != null) {
                    levelListDrawable.addLevel(1, 1, new BitmapDrawable(bitmap));
                    levelListDrawable.setBounds(0, 0, bitmap.getWidth(), bitmap.getHeight());
                    levelListDrawable.setLevel(1);
                    MImageGetter.this.container.invalidate();
                    MImageGetter.this.container.setText(MImageGetter.this.container.getText());
                }
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Bitmap) obj, (Transition<? super Bitmap>) transition);
            }
        });
        return levelListDrawable;
    }
}
