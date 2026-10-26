package com.hismith.hismithapp.utils;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import androidx.core.content.FileProvider;
import java.io.File;

/* loaded from: classes3.dex */
public class FileProviderUtil {
    public static Uri getUriForFile(Context context, File file) {
        return getUriForFile24(context, file);
    }

    public static Uri getUriForFile24(Context context, File file) {
        return FileProvider.getUriForFile(context, context.getPackageName() + ".provider", file);
    }

    public static void setIntentDataAndType(Context context, Intent intent, String str, File file, boolean z) {
        intent.setDataAndType(getUriForFile(context, file), str);
        intent.addFlags(1);
        if (z) {
            intent.addFlags(2);
        }
    }
}
