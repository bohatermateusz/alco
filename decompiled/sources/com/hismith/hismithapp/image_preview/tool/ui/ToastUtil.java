package com.hismith.hismithapp.image_preview.tool.ui;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.widget.Toast;

/* loaded from: classes3.dex */
public class ToastUtil {
    private static final Handler HANDLER = new Handler(Looper.getMainLooper());

    private static class InnerClass {
        private static ToastUtil instance = new ToastUtil();

        private InnerClass() {
        }
    }

    public static ToastUtil getInstance() {
        return InnerClass.instance;
    }

    public void _long(final Context context, final String str) {
        HANDLER.post(new Runnable() { // from class: com.hismith.hismithapp.image_preview.tool.ui.ToastUtil.2
            @Override // java.lang.Runnable
            public void run() {
                Toast.makeText(context.getApplicationContext(), str, 1).show();
            }
        });
    }

    public void _short(final Context context, final String str) {
        HANDLER.post(new Runnable() { // from class: com.hismith.hismithapp.image_preview.tool.ui.ToastUtil.1
            @Override // java.lang.Runnable
            public void run() {
                Toast.makeText(context.getApplicationContext(), str, 0).show();
            }
        });
    }
}
