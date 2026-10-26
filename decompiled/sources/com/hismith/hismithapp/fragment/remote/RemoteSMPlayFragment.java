package com.hismith.hismithapp.fragment.remote;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.graphics.Point;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.animation.ScaleAnimation;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.manager.RemoteDataManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.PlayView_Four;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

/* loaded from: classes3.dex */
public class RemoteSMPlayFragment extends RemoteControlFragment implements View.OnClickListener {
    private float downY;
    private ImageView img_reversal;
    private ConstraintLayout layout_action;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private ImageView mImgHeart;
    private ImageView mImgMode;
    private ImageView mImgVibrate;
    private FrameLayout mLayoutArrowDown;
    private FrameLayout mLayoutArrowUp;
    private ImageView mLayoutBg;
    private GridLayoutManager mLayoutManager;
    private FrameLayout mLayoutMode;
    private LinearLayout mLayoutModeBorder;
    private FrameLayout mLayoutModeBtn;
    private ViewGroup mLayoutSafeSpeed;
    private FrameLayout mLayoutVibrateArrowDown;
    private FrameLayout mLayoutVibrateArrowUp;
    private FrameLayout mLayoutVibrateBtn;
    private FrameLayout mLayoutVibration;
    private LinearLayout mLayoutVibrationBorder;
    private PlayView_Four mPlayView;
    private ProgressBar mProgressSpeed;
    private RecyclerView mRecyclerMode;
    private RecyclerView mRecyclerVibration;
    private TextView mTxtMinSpeed;
    private TextView mTxtSafeSpeed;
    private TextView mTxtSpeed;
    private IplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private int remoteSpeed = 0;
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private int mSafeMax = 100;
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment.7
        /* JADX WARN: Code restructure failed: missing block: B:7:0x000d, code lost:
        
            if (r0 != 3) goto L34;
         */
        @Override // android.view.View.OnTouchListener
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action != 0) {
                if (action != 1) {
                    if (action == 2) {
                        int top = (int) (view.getTop() + (motionEvent.getY() - RemoteSMPlayFragment.this.downY));
                        if (top < 0) {
                            top = 0;
                        } else if (top > RemoteSMPlayFragment.this.mProgressSpeed.getHeight()) {
                            top = RemoteSMPlayFragment.this.mProgressSpeed.getHeight();
                        }
                        int height = view.getHeight() + top;
                        LogUtil.e("onTouch  t:" + top + ", b:" + height + ", offset:" + (height - top));
                        ViewGroup.LayoutParams layoutParams = RemoteSMPlayFragment.this.mLayoutSafeSpeed.getLayoutParams();
                        if (RemoteSMPlayFragment.this.mLayoutSafeSpeed.getLayoutParams() instanceof LinearLayout.LayoutParams) {
                            ((LinearLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
                        } else if (RemoteSMPlayFragment.this.mLayoutSafeSpeed.getLayoutParams() instanceof FrameLayout.LayoutParams) {
                            ((FrameLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
                        }
                        RemoteSMPlayFragment.this.mLayoutSafeSpeed.setLayoutParams(layoutParams);
                        int height2 = 100 - (((int) (top * 100.0d)) / RemoteSMPlayFragment.this.mProgressSpeed.getHeight());
                        RemoteSMPlayFragment.this.mTxtSafeSpeed.setText(height2 + "");
                        RemoteSMPlayFragment.this.mSafeMax = height2;
                        RemoteDataManager.getInstance().setLimitMax(RemoteSMPlayFragment.this.mSafeMax);
                        if (height2 >= 0 && height2 < RemoteSMPlayFragment.this.mProgressSpeed.getSecondaryProgress()) {
                            BleManager.getInstance(RemoteSMPlayFragment.this.baseActivity).writeSpeed(height2);
                            RemoteSMPlayFragment.this.refreshSpeedUI(height2);
                        } else if (height2 >= RemoteSMPlayFragment.this.mProgressSpeed.getSecondaryProgress() && RemoteSMPlayFragment.this.mProgressSpeed.getProgress() != RemoteSMPlayFragment.this.mProgressSpeed.getSecondaryProgress()) {
                            int secondaryProgress = RemoteSMPlayFragment.this.mProgressSpeed.getSecondaryProgress();
                            BleManager.getInstance(RemoteSMPlayFragment.this.baseActivity).writeSpeed(secondaryProgress);
                            RemoteSMPlayFragment.this.refreshSpeedUI(secondaryProgress);
                        }
                    }
                }
                LogUtil.e("onTouch 松开Y:" + motionEvent.getY());
            } else {
                RemoteSMPlayFragment.this.downY = motionEvent.getY();
                LogUtil.e("onTouch 按下Y:" + motionEvent.getY());
                LogUtil.e("onTouch top:" + view.getTop());
            }
            return true;
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public void initModeShow() {
        int i = this.isShowMode;
        if (i != 0) {
            if (i == 1) {
                startAnimation(this.mLayoutMode, this.mLayoutVibration);
                return;
            } else {
                if (i == 2) {
                    startAnimation(this.mLayoutVibration, this.mLayoutMode);
                    return;
                }
                return;
            }
        }
        FrameLayout frameLayout = this.mLayoutMode;
        if (frameLayout != null && frameLayout.getVisibility() == 0) {
            stopAnimation(this.mLayoutMode);
        }
        FrameLayout frameLayout2 = this.mLayoutVibration;
        if (frameLayout2 == null || frameLayout2.getVisibility() != 0) {
            return;
        }
        stopAnimation(this.mLayoutVibration);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.mTxtSafeSpeed.setText(i + "");
        double height = this.mProgressSpeed.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        ViewGroup.LayoutParams layoutParams = this.mLayoutSafeSpeed.getLayoutParams();
        if (this.mLayoutSafeSpeed.getLayoutParams() instanceof LinearLayout.LayoutParams) {
            ((LinearLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        } else if (this.mLayoutSafeSpeed.getLayoutParams() instanceof FrameLayout.LayoutParams) {
            ((FrameLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        }
        this.mLayoutSafeSpeed.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        if (this.mRecyclerMode.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.mLayoutArrowDown.getVisibility() == 4) {
                this.mLayoutArrowDown.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.mLayoutArrowDown.setVisibility(4);
        }
        if (!this.mRecyclerMode.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.mLayoutArrowUp.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.mLayoutArrowUp.getVisibility() == 4) {
                this.mLayoutArrowUp.setVisibility(0);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollVibrateHideImg() {
        if (this.mRecyclerVibration.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.mLayoutVibrateArrowDown.getVisibility() == 4) {
                this.mLayoutVibrateArrowDown.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.mLayoutVibrateArrowDown.setVisibility(4);
        }
        if (!this.mRecyclerVibration.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.mLayoutVibrateArrowUp.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.mLayoutVibrateArrowUp.getVisibility() == 4) {
                this.mLayoutVibrateArrowUp.setVisibility(0);
            }
        }
    }

    private void startAnimation(final FrameLayout frameLayout, final FrameLayout frameLayout2) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 0.0f, 1.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment.4
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteSMPlayFragment.this.inAnim = false;
                if (i != RemoteSMPlayFragment.this.isShowMode) {
                    RemoteSMPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteSMPlayFragment.this.inAnim = true;
                frameLayout.setVisibility(0);
            }
        });
        if (frameLayout2 == null || frameLayout2.getVisibility() != 0) {
            if (frameLayout.getVisibility() != 0) {
                AnimatorSet animatorSet = new AnimatorSet();
                this.mAnimatorSet = animatorSet;
                animatorSet.playTogether(ofFloat, ofFloat2);
                this.mAnimatorSet.start();
                return;
            }
            return;
        }
        ObjectAnimator ofFloat3 = ObjectAnimator.ofFloat(frameLayout2, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat4 = ObjectAnimator.ofFloat(frameLayout2, "scaleY", 1.0f, 0.0f);
        ofFloat3.setDuration(this.animationDuration);
        ofFloat4.setDuration(this.animationDuration);
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment.5
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteSMPlayFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteSMPlayFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet2 = new AnimatorSet();
        this.mAnimatorSet = animatorSet2;
        animatorSet2.play(ofFloat3).with(ofFloat4);
        this.mAnimatorSet.play(ofFloat3).before(ofFloat);
        this.mAnimatorSet.play(ofFloat).with(ofFloat2);
        this.mAnimatorSet.start();
    }

    private void stopAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 1.0f, 0.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment.6
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteSMPlayFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != RemoteSMPlayFragment.this.isShowMode) {
                    RemoteSMPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteSMPlayFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:45:0x0187 -> B:40:0x018a). Please report as a decompilation issue!!! */
    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void init() {
        PlayView_Four playView_Four;
        super.init();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.mRecyclerMode.setLayoutManager(gridLayoutManager);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this.baseActivity, 2);
        this.mVibrationLayoutManager = gridLayoutManager2;
        this.mRecyclerVibration.setLayoutManager(gridLayoutManager2);
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = iplayModeAdapter;
        this.mRecyclerMode.setAdapter(iplayModeAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        this.mRecyclerVibration.setAdapter(iplayModeAdapter2);
        if (BleManager.getInstance(this.baseActivity).getmConnBleDevice() != null) {
            this.mTxtSpeed.setText(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed() + "");
            this.mProgressSpeed.setProgress(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed());
            if (MyApplication.getApplication().getmProductModel() != null && (playView_Four = this.mPlayView) != null) {
                playView_Four.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
            try {
                if (ProductUtil.hasMode(0)) {
                    this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
                    int mode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMode();
                    if (mode > 0) {
                        this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
                        IplayModeAdapter iplayModeAdapter3 = this.mAdapter;
                        if (iplayModeAdapter3 != null) {
                            iplayModeAdapter3.setClickPosition(mode - 1);
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                boolean hasMode = ProductUtil.hasMode(1);
                FrameLayout frameLayout = this.mLayoutVibrateBtn;
                if (hasMode) {
                    frameLayout.setVisibility(0);
                    this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
                    int vibrationMode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibrationMode();
                    ImageView imageView = this.mImgVibrate;
                    if (vibrationMode >= 1) {
                        imageView.setImageResource(R.mipmap.ic_iplay_vibration_select);
                        IplayModeAdapter iplayModeAdapter4 = this.mVibrationAdapter;
                        if (iplayModeAdapter4 != null) {
                            iplayModeAdapter4.setClickPosition(vibrationMode - 1);
                        }
                    } else {
                        imageView.setImageResource(R.mipmap.ic_iplay_vibration);
                    }
                } else {
                    frameLayout.setVisibility(8);
                }
            } catch (Exception e2) {
                e2.printStackTrace();
            }
            try {
                if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getReversal() != 1) {
                    this.img_reversal.clearAnimation();
                    this.img_reversal.setVisibility(8);
                } else {
                    this.img_reversal.clearAnimation();
                    this.img_reversal.setVisibility(0);
                    int reversal = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getReversal();
                    ImageView imageView2 = this.img_reversal;
                    if (reversal == 1) {
                        AnimUtil.rotate(imageView2, 1);
                        this.img_reversal.setRotationY(180.0f);
                    } else {
                        AnimUtil.rotate(imageView2, 0);
                        this.img_reversal.setRotationY(0.0f);
                    }
                }
            } catch (Exception e3) {
                e3.printStackTrace();
            }
        }
        this.mSafeMax = RemoteDataManager.getInstance().getLimitMax();
        scrollModeHideImg();
        scrollVibrateHideImg();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.iplay_layout_mode_btn) {
            if (this.inAnim) {
                return;
            }
            if (this.mLayoutMode.getVisibility() == 0) {
                stopAnimation(this.mLayoutMode);
                return;
            }
            this.isShowMode = 1;
            int visibility = this.mLayoutVibration.getVisibility();
            FrameLayout frameLayout = this.mLayoutMode;
            if (visibility == 0) {
                startAnimation(frameLayout, this.mLayoutVibration);
                return;
            } else {
                startAnimation(frameLayout, null);
                return;
            }
        }
        if (view.getId() == R.id.iplay_layout_vibrate_btn) {
            if (this.inAnim) {
                return;
            }
            if (this.mLayoutVibration.getVisibility() == 0) {
                stopAnimation(this.mLayoutVibration);
                return;
            }
            this.isShowMode = 2;
            int visibility2 = this.mLayoutMode.getVisibility();
            FrameLayout frameLayout2 = this.mLayoutVibration;
            if (visibility2 == 0) {
                startAnimation(frameLayout2, this.mLayoutMode);
                return;
            } else {
                startAnimation(frameLayout2, null);
                return;
            }
        }
        if (view.getId() == R.id.iplay_layout_mode || view.getId() == R.id.iplay_layout_vibration) {
            if (this.mLayoutMode.getVisibility() == 0) {
                stopAnimation(this.mLayoutMode);
            }
            if (this.mLayoutVibration.getVisibility() == 0) {
                stopAnimation(this.mLayoutVibration);
                return;
            }
            return;
        }
        if (view.getId() == R.id.iplay_layout_arrow_up) {
            if (this.mRecyclerMode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.mRecyclerMode.smoothScrollToPosition(0);
            return;
        }
        if (view.getId() == R.id.iplay_layout_arrow_down) {
            if (this.mRecyclerMode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.mRecyclerMode.smoothScrollToPosition(this.mAdapter.getItemCount() - 1);
            return;
        }
        if (view.getId() == R.id.iplay_layout_vibrate_arrow_up) {
            if (this.mRecyclerVibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
                return;
            }
            this.mRecyclerVibration.smoothScrollToPosition(0);
            return;
        }
        if (view.getId() != R.id.iplay_layout_vibrate_arrow_down || this.mRecyclerVibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
            return;
        }
        this.mRecyclerVibration.smoothScrollToPosition(this.mVibrationAdapter.getItemCount() - 1);
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_remote_sm, (ViewGroup) null);
        this.mLayoutBg = (ImageView) inflate.findViewById(R.id.iplay_layout_bg);
        this.mPlayView = (PlayView_Four) inflate.findViewById(R.id.iplay_playView);
        this.mProgressSpeed = (ProgressBar) inflate.findViewById(R.id.iplay_progress_speed);
        this.mLayoutSafeSpeed = (ViewGroup) inflate.findViewById(R.id.iplay_layout_speed_safe);
        this.mTxtSafeSpeed = (TextView) inflate.findViewById(R.id.iplay_txt_speed_safe);
        this.mTxtMinSpeed = (TextView) inflate.findViewById(R.id.iplay_txt_min_speed);
        this.mTxtSpeed = (TextView) inflate.findViewById(R.id.iplay_txt_speed);
        this.mLayoutModeBtn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode_btn);
        this.mLayoutVibrateBtn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_btn);
        this.mImgMode = (ImageView) inflate.findViewById(R.id.iplay_img_mode);
        this.mImgVibrate = (ImageView) inflate.findViewById(R.id.iplay_img_vibrate);
        this.mLayoutMode = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode);
        this.mLayoutModeBorder = (LinearLayout) inflate.findViewById(R.id.iplay_layout_mode_border);
        this.mRecyclerMode = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_mode);
        this.mLayoutVibration = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibration);
        this.mLayoutVibrationBorder = (LinearLayout) inflate.findViewById(R.id.iplay_layout_vibration_border);
        this.mRecyclerVibration = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_vibration);
        this.mImgHeart = (ImageView) inflate.findViewById(R.id.iplay_img_heart);
        this.mLayoutArrowUp = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_up);
        this.mLayoutArrowDown = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_down);
        this.mLayoutVibrateArrowUp = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.mLayoutVibrateArrowDown = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.layout_action = (ConstraintLayout) inflate.findViewById(R.id.layout_action);
        this.img_reversal = (ImageView) inflate.findViewById(R.id.img_reversal);
        this.mPlayView.setProgressBar(this.mProgressSpeed);
        this.mPlayView.setTouch(false);
        this.mPlayView.setMy(false);
        return inflate;
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshAutoUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshLinePoints(Map<String, Point> map) {
        PlayView_Four playView_Four;
        if (map == null || (playView_Four = this.mPlayView) == null) {
            return;
        }
        playView_Four.setPointArr(map);
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshMusicUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshPositionUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshReversalRotation(int i) {
        if (this.img_reversal.getVisibility() == 0) {
            ImageView imageView = this.img_reversal;
            if (i == 1) {
                AnimUtil.rotate(imageView, 1);
                this.img_reversal.setRotationY(180.0f);
            } else {
                AnimUtil.rotate(imageView, 0);
                this.img_reversal.setRotationY(0.0f);
            }
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshSmoothnessUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshSpeedModeUI(int i) {
        try {
            if (i <= 0) {
                this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
            } else {
                this.isShowMode = 1;
                if (this.mLayoutMode.getVisibility() != 0) {
                    startAnimation(this.mLayoutMode, this.mLayoutVibration);
                }
                this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode_select);
            }
            IplayModeAdapter iplayModeAdapter = this.mAdapter;
            if (iplayModeAdapter != null) {
                int i2 = i - 1;
                iplayModeAdapter.setClickPosition(i2);
                if (i > 0) {
                    this.mRecyclerMode.smoothScrollToPosition(i2);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshSpeedUI(int i) {
        try {
            LogUtil.e("refreshSpeedUI:" + i);
            this.mTxtSpeed.setText(i + "");
            this.mProgressSpeed.setProgress(i);
            this.isShowMode = 0;
            if (this.mLayoutMode.getVisibility() == 0) {
                stopAnimation(this.mLayoutMode);
            }
            this.mImgMode.setImageResource(R.mipmap.ic_iplay_mode);
            IplayModeAdapter iplayModeAdapter = this.mAdapter;
            if (iplayModeAdapter != null) {
                iplayModeAdapter.setClickPosition(-1);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshStartPositionUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshStopPositionUi(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshVibrateLinePoints(Map<String, Point> map) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshVibrateModeUI(int i) {
        try {
            if (i <= 0) {
                this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration);
            } else {
                this.isShowMode = 2;
                if (this.mLayoutVibration.getVisibility() != 0) {
                    startAnimation(this.mLayoutVibration, this.mLayoutMode);
                }
                this.mImgVibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
            }
            IplayModeAdapter iplayModeAdapter = this.mVibrationAdapter;
            if (iplayModeAdapter != null) {
                int i2 = i - 1;
                iplayModeAdapter.setClickPosition(i2);
                if (i > 0) {
                    this.mRecyclerVibration.smoothScrollToPosition(i2);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshVibrateUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshWebSpeedUI(int i) {
        ProgressBar progressBar = this.mProgressSpeed;
        if (progressBar != null) {
            progressBar.setSecondaryProgress(i);
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshWebVibrateUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void setListener() {
        super.setListener();
        this.mLayoutSafeSpeed.setOnTouchListener(this.safeOnTouch);
        this.mLayoutModeBtn.setOnClickListener(this);
        this.mLayoutVibrateBtn.setOnClickListener(this);
        this.mRecyclerMode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment.1
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                RemoteSMPlayFragment.this.scrollModeHideImg();
            }
        });
        this.mRecyclerVibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment.2
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                RemoteSMPlayFragment.this.scrollVibrateHideImg();
            }
        });
        this.mLayoutSafeSpeed.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment.3
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                RemoteSMPlayFragment.this.mLayoutSafeSpeed.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                RemoteSMPlayFragment remoteSMPlayFragment = RemoteSMPlayFragment.this;
                remoteSMPlayFragment.refreshSafeUI(remoteSMPlayFragment.mSafeMax);
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void startHeart() {
        ImageView imageView = this.mImgHeart;
        if (imageView != null) {
            imageView.setVisibility(0);
            ScaleAnimation scaleAnimation = new ScaleAnimation(1.0f, 0.7f, 1.0f, 0.7f, 1, 0.5f, 1, 0.5f);
            scaleAnimation.setDuration(250L);
            scaleAnimation.setFillAfter(true);
            scaleAnimation.setRepeatCount(-1);
            scaleAnimation.setRepeatMode(2);
            this.mImgHeart.startAnimation(scaleAnimation);
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void stopHeart() {
        ImageView imageView = this.mImgHeart;
        if (imageView == null || imageView.getVisibility() != 0) {
            return;
        }
        this.mImgHeart.clearAnimation();
        this.mImgHeart.setVisibility(8);
    }
}
