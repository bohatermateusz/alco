package com.hismith.hismithapp.utils;

import android.content.Context;
import androidx.constraintlayout.core.motion.utils.TypedValues;

/* loaded from: classes3.dex */
public class ResUtil {
    public static int getMipmapId(Context context, String str) {
        return context.getResources().getIdentifier(str, "mipmap", context.getPackageName());
    }

    public static int getStringId(Context context, String str) {
        return context.getResources().getIdentifier(str, TypedValues.Custom.S_STRING, context.getPackageName());
    }
}
