package com.hismith.hismithapp.image_preview.view.helper;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.Context;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.ViewConfiguration;
import android.view.animation.LinearInterpolator;
import android.widget.LinearLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.image_preview.view.nine.ViewHelper;
import com.hismith.hismithapp.image_preview.view.photoview.PhotoView;

/* loaded from: classes3.dex */
public class FingerDragHelper extends LinearLayout {
    private static final long DURATION = 200;
    private static final int MAX_EXIT_Y = 500;
    private static int MAX_TRANSLATE_Y = 500;
    private static final String TAG = "FingerDragHelper";
    private int fadeIn;
    private int fadeOut;
    private PhotoView imageGif;
    private SubsamplingScaleImageViewDragClose imageView;
    private boolean isAnimate;
    private float mDownY;
    private float mLastTranslationY;
    private onAlphaChangedListener mOnAlphaChangedListener;
    private int mTouchslop;
    private float mTranslationY;

    public interface onAlphaChangedListener {
        void onTranslationYChanged(MotionEvent motionEvent, float f);
    }

    public FingerDragHelper(Context context) {
        this(context, null);
    }

    public FingerDragHelper(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public FingerDragHelper(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isAnimate = false;
        this.fadeIn = R.anim.fade_in_150;
        this.fadeOut = R.anim.fade_out_150;
        initViews();
    }

    private void initViews() {
        this.mTouchslop = ViewConfiguration.getTouchSlop();
    }

    private void onActionUp() {
        boolean isEnableUpDragClose = ImagePreview.getInstance().isEnableUpDragClose();
        float f = this.mTranslationY;
        if (isEnableUpDragClose) {
            if (Math.abs(f) > 500.0f) {
                exitWithTranslation(this.mTranslationY);
                return;
            } else {
                resetCallBackAnimation();
                return;
            }
        }
        if (f > 500.0f) {
            exitWithTranslation(f);
        } else {
            resetCallBackAnimation();
        }
    }

    private void onOneFingerPanActionMove(MotionEvent motionEvent) {
        float rawY = (motionEvent.getRawY() - this.mDownY) + this.mLastTranslationY;
        this.mTranslationY = rawY;
        onAlphaChangedListener onalphachangedlistener = this.mOnAlphaChangedListener;
        if (onalphachangedlistener != null) {
            onalphachangedlistener.onTranslationYChanged(motionEvent, rawY);
        }
        ViewHelper.setScrollY(this, -((int) this.mTranslationY));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void reset() {
        onAlphaChangedListener onalphachangedlistener = this.mOnAlphaChangedListener;
        if (onalphachangedlistener != null) {
            onalphachangedlistener.onTranslationYChanged(null, this.mTranslationY);
        }
    }

    private void resetCallBackAnimation() {
        ValueAnimator ofFloat = ValueAnimator.ofFloat(this.mTranslationY, 0.0f);
        ofFloat.setDuration(DURATION);
        ofFloat.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper.5
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                if (FingerDragHelper.this.isAnimate) {
                    FingerDragHelper.this.mTranslationY = ((Float) valueAnimator.getAnimatedValue()).floatValue();
                    FingerDragHelper fingerDragHelper = FingerDragHelper.this;
                    fingerDragHelper.mLastTranslationY = fingerDragHelper.mTranslationY;
                    FingerDragHelper fingerDragHelper2 = FingerDragHelper.this;
                    ViewHelper.setScrollY(fingerDragHelper2, -((int) fingerDragHelper2.mTranslationY));
                }
            }
        });
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper.6
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                if (FingerDragHelper.this.isAnimate) {
                    FingerDragHelper.this.mTranslationY = 0.0f;
                    FingerDragHelper.this.invalidate();
                    FingerDragHelper.this.reset();
                }
                FingerDragHelper.this.isAnimate = false;
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                FingerDragHelper.this.isAnimate = true;
            }
        });
        ofFloat.start();
    }

    public void exitWithTranslation(float f) {
        float f2 = this.mTranslationY;
        if (f > 0.0f) {
            ValueAnimator ofFloat = ValueAnimator.ofFloat(f2, getHeight());
            ofFloat.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper.1
                @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                public void onAnimationUpdate(ValueAnimator valueAnimator) {
                    ViewHelper.setScrollY(FingerDragHelper.this, -((int) ((Float) valueAnimator.getAnimatedValue()).floatValue()));
                }
            });
            ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper.2
                @Override // android.animation.Animator.AnimatorListener
                public void onAnimationCancel(Animator animator) {
                }

                @Override // android.animation.Animator.AnimatorListener
                public void onAnimationEnd(Animator animator) {
                    FingerDragHelper.this.reset();
                    Activity activity = (Activity) FingerDragHelper.this.getContext();
                    activity.finish();
                    activity.overridePendingTransition(FingerDragHelper.this.fadeIn, FingerDragHelper.this.fadeOut);
                }

                @Override // android.animation.Animator.AnimatorListener
                public void onAnimationRepeat(Animator animator) {
                }

                @Override // android.animation.Animator.AnimatorListener
                public void onAnimationStart(Animator animator) {
                }
            });
            ofFloat.setDuration(DURATION);
            ofFloat.setInterpolator(new LinearInterpolator());
            ofFloat.start();
            return;
        }
        ValueAnimator ofFloat2 = ValueAnimator.ofFloat(f2, -getHeight());
        ofFloat2.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper.3
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                ViewHelper.setScrollY(FingerDragHelper.this, -((int) ((Float) valueAnimator.getAnimatedValue()).floatValue()));
            }
        });
        ofFloat2.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper.4
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                FingerDragHelper.this.reset();
                ((Activity) FingerDragHelper.this.getContext()).finish();
                ((Activity) FingerDragHelper.this.getContext()).overridePendingTransition(FingerDragHelper.this.fadeIn, FingerDragHelper.this.fadeOut);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
            }
        });
        ofFloat2.setDuration(DURATION);
        ofFloat2.setInterpolator(new LinearInterpolator());
        ofFloat2.start();
    }

    @Override // android.view.View
    protected void onFinishInflate() {
        super.onFinishInflate();
        this.imageView = (SubsamplingScaleImageViewDragClose) getChildAt(0);
        this.imageGif = (PhotoView) getChildAt(1);
    }

    /* JADX WARN: Code restructure failed: missing block: B:3:0x000d, code lost:
    
        if (r0 != 2) goto L38;
     */
    @Override // android.view.ViewGroup
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        int action = motionEvent.getAction() & motionEvent.getActionMasked();
        if (action == 0) {
            this.mDownY = motionEvent.getRawY();
        }
        if (ImagePreview.getInstance().isEnableDragClose()) {
            PhotoView photoView = this.imageGif;
            if (photoView != null && photoView.getVisibility() == 0) {
                return this.imageGif.getScale() <= this.imageGif.getMinimumScale() + 0.001f && (this.imageGif.getMaxTouchCount() == 0 || this.imageGif.getMaxTouchCount() == 1) && Math.abs(motionEvent.getRawY() - this.mDownY) > ((float) (this.mTouchslop * 2));
            }
            SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose = this.imageView;
            if (subsamplingScaleImageViewDragClose != null && subsamplingScaleImageViewDragClose.getVisibility() == 0 && this.imageView.getScale() <= this.imageView.getMinScale() + 0.001f && ((this.imageView.getMaxTouchCount() == 0 || this.imageView.getMaxTouchCount() == 1) && Math.abs(motionEvent.getRawY() - this.mDownY) > this.mTouchslop * 2 && this.imageView.atYEdge)) {
                return true;
            }
        }
        return false;
    }

    /* JADX WARN: Code restructure failed: missing block: B:5:0x000f, code lost:
    
        if (r0 != 2) goto L22;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int action = motionEvent.getAction() & motionEvent.getActionMasked();
        if (action != 0) {
            if (action == 1) {
                onActionUp();
            }
            return true;
        }
        this.mDownY = motionEvent.getRawY();
        if (ImagePreview.getInstance().isEnableDragClose()) {
            PhotoView photoView = this.imageGif;
            if (photoView == null || photoView.getVisibility() != 0) {
                SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose = this.imageView;
                if (subsamplingScaleImageViewDragClose != null && subsamplingScaleImageViewDragClose.getVisibility() == 0) {
                    onOneFingerPanActionMove(motionEvent);
                }
            } else {
                onOneFingerPanActionMove(motionEvent);
            }
        }
        return true;
    }

    public void setOnAlphaChangeListener(onAlphaChangedListener onalphachangedlistener) {
        this.mOnAlphaChangedListener = onalphachangedlistener;
    }
}
