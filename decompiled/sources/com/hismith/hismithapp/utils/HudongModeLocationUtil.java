package com.hismith.hismithapp.utils;

import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.ConstraintSet;

/* loaded from: classes3.dex */
public class HudongModeLocationUtil {
    public static void initModePosition(String str, ConstraintLayout constraintLayout, int i, int i2, int i3, int i4) {
        if (str != null) {
            if (str.contains("01device")) {
                ConstraintSet constraintSet = new ConstraintSet();
                constraintSet.clone(constraintLayout);
                if (i2 > 0) {
                    constraintSet.constrainPercentWidth(i2, 0.068f);
                    constraintSet.setHorizontalBias(i2, Math.abs(i4 - 0.985f));
                    constraintSet.setVerticalBias(i2, 0.5f);
                }
                constraintSet.applyTo(constraintLayout);
                return;
            }
            if (str.contains("02device") || str.contains("03device") || str.contains("05device")) {
                ConstraintSet constraintSet2 = new ConstraintSet();
                constraintSet2.clone(constraintLayout);
                if (i2 > 0) {
                    constraintSet2.constrainPercentWidth(i2, 0.1f);
                    constraintSet2.setHorizontalBias(i2, 0.5f);
                    constraintSet2.setVerticalBias(i2, 0.95f);
                }
                constraintSet2.applyTo(constraintLayout);
                return;
            }
            if (str.contains("09device")) {
                ConstraintSet constraintSet3 = new ConstraintSet();
                constraintSet3.clone(constraintLayout);
                if (i2 > 0) {
                    constraintSet3.constrainPercentWidth(i2, 0.06f);
                    constraintSet3.setHorizontalBias(i2, 0.15f);
                    constraintSet3.setVerticalBias(i2, 0.85f);
                    constraintSet3.connect(i2, 3, i, 3);
                    constraintSet3.connect(i2, 4, i, 4);
                    constraintSet3.connect(i2, 6, i, 6);
                    constraintSet3.connect(i2, 7, i, 7);
                }
                if (i3 > 0) {
                    constraintSet3.constrainPercentWidth(i3, 0.06f);
                    constraintSet3.setHorizontalBias(i3, 0.85f);
                    constraintSet3.setVerticalBias(i3, 0.85f);
                    constraintSet3.connect(i3, 3, i, 3);
                    constraintSet3.connect(i3, 4, i, 4);
                    constraintSet3.connect(i3, 6, i, 6);
                    constraintSet3.connect(i3, 7, i, 7);
                }
                constraintSet3.applyTo(constraintLayout);
                return;
            }
            if (str.contains("08device")) {
                ConstraintSet constraintSet4 = new ConstraintSet();
                constraintSet4.clone(constraintLayout);
                if (i2 > 0) {
                    constraintSet4.constrainPercentWidth(i2, 0.06f);
                    constraintSet4.setHorizontalBias(i2, 0.22f);
                    constraintSet4.setVerticalBias(i2, 0.35f);
                    constraintSet4.connect(i2, 3, i, 3);
                    constraintSet4.connect(i2, 4, i, 4);
                    constraintSet4.connect(i2, 6, i, 6);
                    constraintSet4.connect(i2, 7, i, 7);
                }
                if (i3 > 0) {
                    constraintSet4.constrainPercentWidth(i3, 0.06f);
                    constraintSet4.setHorizontalBias(i3, 0.78f);
                    constraintSet4.setVerticalBias(i3, 0.35f);
                    constraintSet4.connect(i3, 3, i, 3);
                    constraintSet4.connect(i3, 4, i, 4);
                    constraintSet4.connect(i3, 6, i, 6);
                    constraintSet4.connect(i3, 7, i, 7);
                }
                constraintSet4.applyTo(constraintLayout);
                return;
            }
            if (str.contains("06device")) {
                ConstraintSet constraintSet5 = new ConstraintSet();
                constraintSet5.clone(constraintLayout);
                if (i2 > 0) {
                    constraintSet5.constrainPercentWidth(i2, 0.1f);
                    constraintSet5.setHorizontalBias(i2, Math.abs(i4 - 0.5f));
                    constraintSet5.setVerticalBias(i2, 0.95f);
                }
                if (i3 > 0) {
                    constraintSet5.constrainPercentWidth(i3, 0.1f);
                    constraintSet5.setHorizontalBias(i3, Math.abs(i4 - 0.5f));
                    constraintSet5.setVerticalBias(i3, 0.5f);
                    if (i4 == 0) {
                        constraintSet5.connect(i3, 7, i, 6);
                        constraintSet5.clear(i3, 6);
                    } else {
                        constraintSet5.connect(i3, 6, i, 7);
                        constraintSet5.clear(i3, 7);
                    }
                    constraintSet5.connect(i3, 3, i, 3);
                    constraintSet5.clear(i3, 4);
                }
                constraintSet5.applyTo(constraintLayout);
                return;
            }
            if (str.contains("04device")) {
                ConstraintSet constraintSet6 = new ConstraintSet();
                constraintSet6.clone(constraintLayout);
                if (i2 > 0) {
                    constraintSet6.constrainPercentWidth(i2, 0.06f);
                    constraintSet6.setHorizontalBias(i2, 0.18f);
                    constraintSet6.setVerticalBias(i2, 0.0f);
                    constraintSet6.connect(i2, 3, i, 3);
                    constraintSet6.connect(i2, 4, i, 4);
                    constraintSet6.connect(i2, 6, i, 6);
                    constraintSet6.connect(i2, 7, i, 7);
                }
                if (i3 > 0) {
                    constraintSet6.constrainPercentWidth(i3, 0.06f);
                    constraintSet6.setHorizontalBias(i3, 0.9f);
                    constraintSet6.setVerticalBias(i3, 0.42f);
                    constraintSet6.connect(i3, 3, i, 3);
                    constraintSet6.connect(i3, 4, i, 4);
                    constraintSet6.connect(i3, 6, i, 6);
                    constraintSet6.connect(i3, 7, i, 7);
                }
                constraintSet6.applyTo(constraintLayout);
                return;
            }
            if (str.contains("07device")) {
                ConstraintSet constraintSet7 = new ConstraintSet();
                constraintSet7.clone(constraintLayout);
                if (i2 > 0) {
                    constraintSet7.constrainPercentWidth(i2, 0.06f);
                    constraintSet7.setHorizontalBias(i2, 0.5f);
                    constraintSet7.setVerticalBias(i2, 0.98f);
                    constraintSet7.connect(i2, 3, i, 3);
                    constraintSet7.connect(i2, 4, i, 4);
                    constraintSet7.connect(i2, 6, i, 6);
                    constraintSet7.connect(i2, 7, i, 7);
                }
                if (i3 > 0) {
                    constraintSet7.constrainPercentWidth(i3, 0.06f);
                    constraintSet7.setHorizontalBias(i3, 0.5f);
                    constraintSet7.setVerticalBias(i3, 0.44f);
                    constraintSet7.connect(i3, 3, i, 3);
                    constraintSet7.connect(i3, 4, i, 4);
                    constraintSet7.connect(i3, 6, i, 6);
                    constraintSet7.connect(i3, 7, i, 7);
                }
                constraintSet7.applyTo(constraintLayout);
            }
        }
    }
}
