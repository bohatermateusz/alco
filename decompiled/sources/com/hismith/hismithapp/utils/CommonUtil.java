package com.hismith.hismithapp.utils;

import android.view.View;

/* loaded from: classes3.dex */
public class CommonUtil {
    public static void measureWidthAndHeight(View view) {
        view.measure(View.MeasureSpec.makeMeasureSpec(0, 0), View.MeasureSpec.makeMeasureSpec(0, 0));
    }
}
