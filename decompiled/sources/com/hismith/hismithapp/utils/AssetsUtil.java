package com.hismith.hismithapp.utils;

import android.content.Context;
import java.io.IOException;
import java.io.InputStream;

/* loaded from: classes3.dex */
public class AssetsUtil {
    public static boolean isAssetExists(Context context, String str) {
        InputStream inputStream = null;
        try {
            try {
                try {
                    InputStream open = context.getAssets().open(str);
                    if (open == null) {
                        open.close();
                        return false;
                    }
                    try {
                        open.close();
                        return true;
                    } catch (IOException e) {
                        e.printStackTrace();
                        return true;
                    }
                } catch (IOException e2) {
                    e2.printStackTrace();
                    return false;
                }
            } catch (IOException e3) {
                e3.printStackTrace();
                inputStream.close();
                throw null;
            }
        } catch (Throwable th) {
            try {
                inputStream.close();
                throw null;
            } catch (IOException e4) {
                e4.printStackTrace();
                throw th;
            }
        }
    }
}
