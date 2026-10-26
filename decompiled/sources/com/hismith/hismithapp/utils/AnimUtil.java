package com.hismith.hismithapp.utils;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.view.animation.LinearInterpolator;
import android.view.animation.RotateAnimation;
import android.view.animation.ScaleAnimation;
import android.view.animation.TranslateAnimation;
import android.widget.FrameLayout;
import android.widget.ImageView;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class AnimUtil {
    private static ValueAnimator valueAnimator;
    private boolean inAnim = false;

    public interface onValueCallback {
        void onCancel();

        void onEnd();

        void onStart();

        void onValue(float f);
    }

    public static void cancelValueAnimation() {
        ValueAnimator valueAnimator2 = valueAnimator;
        if (valueAnimator2 != null) {
            valueAnimator2.cancel();
        }
    }

    public static void pauseValueAnimation() {
        ValueAnimator valueAnimator2 = valueAnimator;
        if (valueAnimator2 != null) {
            valueAnimator2.pause();
        }
    }

    public static void resumeValueAnimation() {
        ValueAnimator valueAnimator2 = valueAnimator;
        if (valueAnimator2 != null) {
            valueAnimator2.resume();
        }
    }

    public static void rotate(View view) {
        rotate(view, 0);
    }

    public static void rotate(View view, int i) {
        float f;
        float f2;
        view.clearAnimation();
        if (i == 1) {
            f2 = 0.0f;
            f = 360.0f;
        } else {
            f = 0.0f;
            f2 = 360.0f;
        }
        RotateAnimation rotateAnimation = new RotateAnimation(f, f2, 1, 0.5f, 1, 0.5f);
        rotateAnimation.setInterpolator(new LinearInterpolator());
        rotateAnimation.setDuration(1000L);
        rotateAnimation.setFillAfter(true);
        rotateAnimation.setRepeatCount(-1);
        view.startAnimation(rotateAnimation);
    }

    public static void scaleGone(final View view) {
        LogUtil.e("scaleGone scaleGone");
        view.clearAnimation();
        LogUtil.e("scaleGone clearAnimation");
        ValueAnimator ofInt = ValueAnimator.ofInt(view.getWidth(), 0);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(500L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.3
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                ViewGroup.LayoutParams layoutParams = view.getLayoutParams();
                layoutParams.width = ((Integer) valueAnimator2.getAnimatedValue()).intValue();
                LogUtil.e("scaleGone onAnimationUpdate:" + valueAnimator2.getAnimatedValue());
                view.setLayoutParams(layoutParams);
            }
        });
        ofInt.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.4
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                LogUtil.e("scaleGone onAnimationEnd");
                view.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
            }
        });
        ofInt.start();
    }

    public static void scaleVisible(final View view, int i) {
        LogUtil.e("scaleVisible scaleVisible");
        view.clearAnimation();
        LogUtil.e("scaleVisible clearAnimation");
        ValueAnimator ofInt = ValueAnimator.ofInt(view.getWidth(), i);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(500L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.5
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                ViewGroup.LayoutParams layoutParams = view.getLayoutParams();
                layoutParams.width = ((Integer) valueAnimator2.getAnimatedValue()).intValue();
                LogUtil.e("scaleVisible onAnimationUpdate" + valueAnimator2.getAnimatedValue());
                view.setLayoutParams(layoutParams);
            }
        });
        ofInt.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.6
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                view.setVisibility(0);
                LogUtil.e("scaleVisible onAnimationStart");
            }
        });
        ofInt.start();
    }

    public static void shakeAnimation(Context context, View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(context, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }

    private void startAnimation(final FrameLayout frameLayout, final FrameLayout frameLayout2, int i) {
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 0.0f, 1.0f);
        long j = i;
        ofFloat.setDuration(j);
        ofFloat2.setDuration(j);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.7
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                AnimUtil.this.inAnim = false;
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                AnimUtil.this.inAnim = true;
                frameLayout.setVisibility(0);
            }
        });
        if (frameLayout2 == null || frameLayout2.getVisibility() != 0) {
            AnimatorSet animatorSet = new AnimatorSet();
            animatorSet.playTogether(ofFloat, ofFloat2);
            animatorSet.start();
            return;
        }
        ObjectAnimator ofFloat3 = ObjectAnimator.ofFloat(frameLayout2, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat4 = ObjectAnimator.ofFloat(frameLayout2, "scaleY", 1.0f, 0.0f);
        ofFloat3.setDuration(j);
        ofFloat4.setDuration(j);
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.8
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                AnimUtil.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                AnimUtil.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet2 = new AnimatorSet();
        animatorSet2.play(ofFloat3).with(ofFloat4);
        animatorSet2.play(ofFloat3).before(ofFloat);
        animatorSet2.play(ofFloat).with(ofFloat2);
        animatorSet2.start();
    }

    public static void startHeart(ImageView imageView) {
        if (imageView != null) {
            imageView.setVisibility(0);
            ScaleAnimation scaleAnimation = new ScaleAnimation(1.0f, 0.7f, 1.0f, 0.7f, 1, 0.5f, 1, 0.5f);
            scaleAnimation.setDuration(250L);
            scaleAnimation.setFillAfter(true);
            scaleAnimation.setRepeatCount(-1);
            scaleAnimation.setRepeatMode(2);
            imageView.startAnimation(scaleAnimation);
        }
    }

    private void stopAnimation(final FrameLayout frameLayout, int i) {
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 1.0f, 0.0f);
        long j = i;
        ofFloat.setDuration(j);
        ofFloat2.setDuration(j);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.9
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                AnimUtil.this.inAnim = false;
                frameLayout.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                AnimUtil.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        animatorSet.playTogether(ofFloat, ofFloat2);
        animatorSet.start();
    }

    public static void stopHeart(ImageView imageView) {
        if (imageView == null || imageView.getVisibility() != 0) {
            return;
        }
        imageView.clearAnimation();
        imageView.setVisibility(8);
    }

    public static void translate(Context context, View view) {
        TranslateAnimation translateAnimation = new TranslateAnimation(0.0f, 0.0f, 0.0f, MeasureUtil.dip2px(context, 10.0f));
        translateAnimation.setDuration(500L);
        translateAnimation.setRepeatCount(-1);
        translateAnimation.setRepeatMode(2);
        view.startAnimation(translateAnimation);
    }

    public static void valueOfFloatAnimation(float f, float f2, final onValueCallback onvaluecallback) {
        ValueAnimator ofFloat = ValueAnimator.ofFloat(f, f2);
        valueAnimator = ofFloat;
        ofFloat.setInterpolator(new LinearInterpolator());
        valueAnimator.setDuration((long) (f2 * 1000.0f));
        valueAnimator.setRepeatCount(0);
        valueAnimator.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.1
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
                onValueCallback onvaluecallback2 = onValueCallback.this;
                if (onvaluecallback2 != null) {
                    onvaluecallback2.onCancel();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                onValueCallback onvaluecallback2 = onValueCallback.this;
                if (onvaluecallback2 != null) {
                    onvaluecallback2.onEnd();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                onValueCallback onvaluecallback2 = onValueCallback.this;
                if (onvaluecallback2 != null) {
                    onvaluecallback2.onStart();
                }
            }
        });
        valueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.utils.AnimUtil.2
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                onValueCallback onvaluecallback2 = onValueCallback.this;
                if (onvaluecallback2 != null) {
                    onvaluecallback2.onValue(((Float) valueAnimator2.getAnimatedValue()).floatValue());
                }
            }
        });
        valueAnimator.start();
    }
}
