package com.hismith.hismithapp.image_preview.glide;

import android.content.Context;
import androidx.appcompat.app.AppCompatActivity;
import com.bumptech.glide.Glide;
import com.bumptech.glide.disklrucache.DiskLruCache;
import com.bumptech.glide.load.engine.cache.DiskCache;
import com.bumptech.glide.load.model.GlideUrl;
import com.bumptech.glide.signature.EmptySignature;
import com.hismith.hismithapp.image_preview.glide.cache.DataCacheKey;
import com.hismith.hismithapp.image_preview.glide.cache.SafeKeyGenerator;
import java.io.File;

/* loaded from: classes.dex */
public class ImageLoader {
    public static void cleanDiskCache(final Context context) {
        new Thread(new Runnable() { // from class: com.hismith.hismithapp.image_preview.glide.ImageLoader.1
            @Override // java.lang.Runnable
            public void run() {
                Glide.get(context.getApplicationContext()).clearDiskCache();
            }
        }).start();
    }

    public static void clearMemory(AppCompatActivity appCompatActivity) {
        Glide.get(appCompatActivity.getApplicationContext()).clearMemory();
    }

    public static File getGlideCacheFile(Context context, String str) {
        try {
            DiskLruCache.Value value = DiskLruCache.open(new File(context.getCacheDir(), DiskCache.Factory.DEFAULT_DISK_CACHE_DIR), 1, 1, 262144000L).get(new SafeKeyGenerator().getSafeKey(new DataCacheKey(new GlideUrl(str), EmptySignature.obtain())));
            if (value != null) {
                return value.getFile(0);
            }
            return null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
