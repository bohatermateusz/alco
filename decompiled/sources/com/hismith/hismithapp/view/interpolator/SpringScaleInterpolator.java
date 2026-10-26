package com.hismith.hismithapp.view.interpolator;

import android.view.animation.Interpolator;

/* loaded from: classes3.dex */
public class SpringScaleInterpolator implements Interpolator {
    private float factor;

    public SpringScaleInterpolator(float f) {
        this.factor = f;
    }

    @Override // android.animation.TimeInterpolator
    public float getInterpolation(float f) {
        return (float) ((Math.pow(2.0d, (-10.0f) * f) * Math.sin(((f - (r6 / 4.0f)) * 6.283185307179586d) / this.factor)) + 1.0d);
    }
}
