package com.hismith.hismithapp.picker;

import android.content.Context;
import android.widget.ImageView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.load.resource.bitmap.CenterCrop;
import com.bumptech.glide.load.resource.bitmap.RoundedCorners;
import com.luck.picture.lib.engine.ImageEngine;
import com.luck.picture.lib.utils.ActivityCompatHelper;

/* loaded from: classes3.dex */
public class GlideEngine implements ImageEngine {

    private static final class InstanceHolder {
        static final GlideEngine instance = new GlideEngine();

        private InstanceHolder() {
        }
    }

    private GlideEngine() {
    }

    public static GlideEngine createGlideEngine() {
        return InstanceHolder.instance;
    }

    @Override // com.luck.picture.lib.engine.ImageEngine
    public void loadAlbumCover(Context context, String str, ImageView imageView) {
        if (ActivityCompatHelper.assertValidRequest(context)) {
            Glide.with(context).asBitmap().load(str).override(180, 180).sizeMultiplier(0.5f).transform(new CenterCrop(), new RoundedCorners(8)).into(imageView);
        }
    }

    @Override // com.luck.picture.lib.engine.ImageEngine
    public void loadGridImage(Context context, String str, ImageView imageView) {
        if (ActivityCompatHelper.assertValidRequest(context)) {
            Glide.with(context).load(str).override(200, 200).centerCrop().into(imageView);
        }
    }

    @Override // com.luck.picture.lib.engine.ImageEngine
    public void loadImage(Context context, ImageView imageView, String str, int i, int i2) {
        if (ActivityCompatHelper.assertValidRequest(context)) {
            Glide.with(context).load(str).override(i, i2).into(imageView);
        }
    }

    @Override // com.luck.picture.lib.engine.ImageEngine
    public void loadImage(Context context, String str, ImageView imageView) {
        if (ActivityCompatHelper.assertValidRequest(context)) {
            Glide.with(context).load(str).into(imageView);
        }
    }

    @Override // com.luck.picture.lib.engine.ImageEngine
    public void pauseRequests(Context context) {
        if (ActivityCompatHelper.assertValidRequest(context)) {
            Glide.with(context).pauseRequests();
        }
    }

    @Override // com.luck.picture.lib.engine.ImageEngine
    public void resumeRequests(Context context) {
        if (ActivityCompatHelper.assertValidRequest(context)) {
            Glide.with(context).resumeRequests();
        }
    }
}
