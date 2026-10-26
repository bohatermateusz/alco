package com.hismith.hismithapp.utils;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.LinearInterpolator;

/* loaded from: classes3.dex */
public class AnimUtiler {
    private int duration = 300;
    private Context mContext;
    private View mView;
    private int mWidth;
    private ValueAnimator valueAnimator;

    public AnimUtiler(Context context, View view, int i) {
        this.mContext = context;
        this.mView = view;
        this.mWidth = i;
    }

    private void cancelAnimator() {
        LogUtil.e("cancelAnimator()");
        ValueAnimator valueAnimator = this.valueAnimator;
        if (valueAnimator == null || !valueAnimator.isRunning()) {
            return;
        }
        LogUtil.e("valueAnimator.cancel()");
        this.valueAnimator.cancel();
    }

    public void scaleGone() {
        cancelAnimator();
        LogUtil.e("scaleGone scaleGone");
        ValueAnimator ofInt = ValueAnimator.ofInt(this.mView.getWidth(), 0);
        this.valueAnimator = ofInt;
        ofInt.setInterpolator(new LinearInterpolator());
        this.valueAnimator.setDuration(this.duration);
        this.valueAnimator.setRepeatCount(0);
        this.valueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.utils.AnimUtiler.1
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                ViewGroup.LayoutParams layoutParams = AnimUtiler.this.mView.getLayoutParams();
                layoutParams.width = ((Integer) valueAnimator.getAnimatedValue()).intValue();
                LogUtil.e("scaleGone onAnimationUpdate:" + valueAnimator.getAnimatedValue());
                AnimUtiler.this.mView.setLayoutParams(layoutParams);
            }
        });
        this.valueAnimator.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtiler.2
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
                LogUtil.e("scaleGone onAnimationCancel");
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                LogUtil.e("scaleGone onAnimationEnd");
                AnimUtiler.this.mView.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
                LogUtil.e("scaleGone onAnimationRepeat");
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                LogUtil.e("scaleGone onAnimationStart");
            }
        });
        this.valueAnimator.start();
    }

    public void scaleVisible() {
        cancelAnimator();
        LogUtil.e("scaleVisible scaleVisible");
        ValueAnimator ofInt = ValueAnimator.ofInt(this.mView.getWidth(), this.mWidth);
        this.valueAnimator = ofInt;
        ofInt.setInterpolator(new LinearInterpolator());
        this.valueAnimator.setDuration(this.duration);
        this.valueAnimator.setRepeatCount(0);
        this.valueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.utils.AnimUtiler.3
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                ViewGroup.LayoutParams layoutParams = AnimUtiler.this.mView.getLayoutParams();
                layoutParams.width = ((Integer) valueAnimator.getAnimatedValue()).intValue();
                LogUtil.e("scaleVisible onAnimationUpdate:" + valueAnimator.getAnimatedValue());
                AnimUtiler.this.mView.setLayoutParams(layoutParams);
            }
        });
        this.valueAnimator.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtiler.4
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
                LogUtil.e("scaleVisible onAnimationCancel");
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                LogUtil.e("scaleVisible onAnimationEnd");
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
                LogUtil.e("scaleVisible onAnimationRepeat");
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                AnimUtiler.this.mView.setVisibility(0);
                LogUtil.e("scaleVisible onAnimationStart");
            }
        });
        this.valueAnimator.start();
    }
}
