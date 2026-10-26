package com.hismith.hismithapp.view.interpolator;

import android.view.animation.Interpolator;

/* loaded from: classes3.dex */
public class AccelerateInterpolator implements Interpolator {
    private float factor;

    public AccelerateInterpolator(float f) {
        this.factor = f;
    }

    @Override // android.animation.TimeInterpolator
    public float getInterpolation(float f) {
        return this.factor == 1.0f ? f * f : (float) Math.pow(f, r2 * 2.0f);
    }
}
