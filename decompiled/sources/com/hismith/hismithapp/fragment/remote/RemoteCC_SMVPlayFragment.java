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
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.manager.RemoteDataManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

/* loaded from: classes3.dex */
public class RemoteCC_SMVPlayFragment extends RemoteControlFragment implements View.OnClickListener {
    private float downY;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_vibrate;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
    private ImageView iplay_layout_bg;
    private FrameLayout iplay_layout_mode;
    private LinearLayout iplay_layout_mode_border;
    private ViewGroup iplay_layout_speed_safe;
    private FrameLayout iplay_layout_vibrate_arrow_down;
    private FrameLayout iplay_layout_vibrate_arrow_up;
    private ViewGroup iplay_layout_vibrate_safe;
    private FrameLayout iplay_layout_vibration;
    private LinearLayout iplay_layout_vibration_border;
    private PlayView_Four iplay_playView;
    private PlayView_Four iplay_playView_vibrate;
    private VerticalProgressBar iplay_progress_speed;
    private VerticalProgressBar iplay_progress_vibrate;
    private RecyclerView iplay_recycler_mode;
    private RecyclerView iplay_recycler_vibration;
    private TextView iplay_txt_speed;
    private TextView iplay_txt_speed_safe;
    private TextView iplay_txt_vibrate;
    private TextView iplay_txt_vibrate_safe;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private GridLayoutManager mLayoutManager;
    private IplayModeAdapter mVibrationAdapter;
    private GridLayoutManager mVibrationLayoutManager;
    private float vibrateDownY;
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private int mSpeedSafeMax = 100;
    private View.OnTouchListener speedSafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.8
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                RemoteCC_SMVPlayFragment.this.downY = motionEvent.getY();
                LogUtil.e("onTouch Down:" + RemoteCC_SMVPlayFragment.this.downY);
                return true;
            }
            if (action != 2) {
                return true;
            }
            int top = (int) (view.getTop() + (motionEvent.getY() - RemoteCC_SMVPlayFragment.this.downY));
            if (top < 0) {
                top = 0;
            } else if (top > RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getHeight()) {
                top = RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getHeight();
            }
            ViewGroup.LayoutParams layoutParams = RemoteCC_SMVPlayFragment.this.iplay_layout_speed_safe.getLayoutParams();
            if (RemoteCC_SMVPlayFragment.this.iplay_layout_speed_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
                ((LinearLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            } else if (RemoteCC_SMVPlayFragment.this.iplay_layout_speed_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
                ((FrameLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            }
            RemoteCC_SMVPlayFragment.this.iplay_layout_speed_safe.setLayoutParams(layoutParams);
            int height = 100 - (((int) (top * 100.0d)) / RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getHeight());
            RemoteCC_SMVPlayFragment.this.iplay_txt_speed_safe.setText("" + height);
            RemoteCC_SMVPlayFragment.this.mSpeedSafeMax = height;
            RemoteDataManager.getInstance().setLimitMax(RemoteCC_SMVPlayFragment.this.mSpeedSafeMax);
            if (height >= 0 && height < RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getSecondaryProgress()) {
                BleManager.getInstance(RemoteCC_SMVPlayFragment.this.baseActivity).writeSpeed(height);
                RemoteCC_SMVPlayFragment.this.refreshSpeedUI(height);
                return true;
            }
            if (height < RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getSecondaryProgress() || RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getProgress() == RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getSecondaryProgress()) {
                return true;
            }
            int secondaryProgress = RemoteCC_SMVPlayFragment.this.iplay_progress_speed.getSecondaryProgress();
            BleManager.getInstance(RemoteCC_SMVPlayFragment.this.baseActivity).writeSpeed(secondaryProgress);
            RemoteCC_SMVPlayFragment.this.refreshSpeedUI(secondaryProgress);
            return true;
        }
    };
    private int mVibrateSafeMax = 100;
    private View.OnTouchListener vibrateSafeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.9
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                RemoteCC_SMVPlayFragment.this.vibrateDownY = motionEvent.getY();
                return true;
            }
            if (action != 2) {
                return true;
            }
            int top = (int) (view.getTop() + (motionEvent.getY() - RemoteCC_SMVPlayFragment.this.vibrateDownY));
            if (top < 0) {
                top = 0;
            } else if (top > RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getHeight()) {
                top = RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getHeight();
            }
            ViewGroup.LayoutParams layoutParams = RemoteCC_SMVPlayFragment.this.iplay_layout_vibrate_safe.getLayoutParams();
            if (RemoteCC_SMVPlayFragment.this.iplay_layout_vibrate_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
                ((LinearLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            } else if (RemoteCC_SMVPlayFragment.this.iplay_layout_vibrate_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
                ((FrameLayout.LayoutParams) layoutParams).setMargins(0, top, 0, 0);
            }
            RemoteCC_SMVPlayFragment.this.iplay_layout_vibrate_safe.setLayoutParams(layoutParams);
            int height = 100 - (((int) (top * 100.0d)) / RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getHeight());
            RemoteCC_SMVPlayFragment.this.iplay_txt_vibrate_safe.setText("" + height);
            RemoteCC_SMVPlayFragment.this.mVibrateSafeMax = height;
            RemoteDataManager.getInstance().setLimitVibrateMax(RemoteCC_SMVPlayFragment.this.mVibrateSafeMax);
            if (height >= 0 && height < RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getSecondaryProgress()) {
                BleManager.getInstance(RemoteCC_SMVPlayFragment.this.baseActivity).writeVibrate(height);
                RemoteCC_SMVPlayFragment.this.refreshVibrateUI(height);
                return true;
            }
            if (height < RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getSecondaryProgress() || RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getProgress() == RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getSecondaryProgress()) {
                return true;
            }
            int secondaryProgress = RemoteCC_SMVPlayFragment.this.iplay_progress_vibrate.getSecondaryProgress();
            BleManager.getInstance(RemoteCC_SMVPlayFragment.this.baseActivity).writeVibrate(secondaryProgress);
            RemoteCC_SMVPlayFragment.this.refreshVibrateUI(secondaryProgress);
            return true;
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public void initModeShow() {
        int i = this.isShowMode;
        if (i != 0) {
            if (i == 1) {
                startAnimation(this.iplay_layout_mode, this.iplay_layout_vibration);
                return;
            } else {
                if (i == 2) {
                    startAnimation(this.iplay_layout_vibration, this.iplay_layout_mode);
                    return;
                }
                return;
            }
        }
        FrameLayout frameLayout = this.iplay_layout_mode;
        if (frameLayout != null && frameLayout.getVisibility() == 0) {
            stopAnimation(this.iplay_layout_mode);
        }
        FrameLayout frameLayout2 = this.iplay_layout_vibration;
        if (frameLayout2 == null || frameLayout2.getVisibility() != 0) {
            return;
        }
        stopAnimation(this.iplay_layout_vibration);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.iplay_txt_speed_safe.setText(i + "");
        double height = this.iplay_progress_speed.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        ViewGroup.LayoutParams layoutParams = this.iplay_layout_speed_safe.getLayoutParams();
        if (this.iplay_layout_speed_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
            ((LinearLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        } else if (this.iplay_layout_speed_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
            ((FrameLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        }
        this.iplay_layout_speed_safe.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshVibrateSafeUI(int i) {
        this.iplay_txt_vibrate_safe.setText(i + "");
        double height = this.iplay_progress_vibrate.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        ViewGroup.LayoutParams layoutParams = this.iplay_layout_vibrate_safe.getLayoutParams();
        if (this.iplay_layout_vibrate_safe.getLayoutParams() instanceof LinearLayout.LayoutParams) {
            ((LinearLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        } else if (this.iplay_layout_vibrate_safe.getLayoutParams() instanceof FrameLayout.LayoutParams) {
            ((FrameLayout.LayoutParams) layoutParams).setMargins(0, i2, 0, 0);
        }
        this.iplay_layout_vibrate_safe.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        if (this.iplay_recycler_mode.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.iplay_layout_arrow_down.getVisibility() == 4) {
                this.iplay_layout_arrow_down.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.iplay_layout_arrow_down.setVisibility(4);
        }
        if (!this.iplay_recycler_mode.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.iplay_layout_arrow_up.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.iplay_layout_arrow_up.getVisibility() == 4) {
                this.iplay_layout_arrow_up.setVisibility(0);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollVibrateHideImg() {
        if (this.iplay_recycler_vibration.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.iplay_layout_vibrate_arrow_down.getVisibility() == 4) {
                this.iplay_layout_vibrate_arrow_down.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.iplay_layout_vibrate_arrow_down.setVisibility(4);
        }
        if (!this.iplay_recycler_vibration.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.iplay_layout_vibrate_arrow_up.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.iplay_layout_vibrate_arrow_up.getVisibility() == 4) {
                this.iplay_layout_vibrate_arrow_up.setVisibility(0);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.5
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteCC_SMVPlayFragment.this.inAnim = false;
                if (i != RemoteCC_SMVPlayFragment.this.isShowMode) {
                    RemoteCC_SMVPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteCC_SMVPlayFragment.this.inAnim = true;
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
        ofFloat3.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.6
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteCC_SMVPlayFragment.this.inAnim = false;
                frameLayout2.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteCC_SMVPlayFragment.this.inAnim = true;
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.7
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteCC_SMVPlayFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != RemoteCC_SMVPlayFragment.this.isShowMode) {
                    RemoteCC_SMVPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteCC_SMVPlayFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void init() {
        super.init();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.iplay_recycler_mode.setLayoutManager(gridLayoutManager);
        GridLayoutManager gridLayoutManager2 = new GridLayoutManager(this.baseActivity, 2);
        this.mVibrationLayoutManager = gridLayoutManager2;
        this.iplay_recycler_vibration.setLayoutManager(gridLayoutManager2);
        this.mModes = new ArrayList();
        this.mVibrationModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = iplayModeAdapter;
        this.iplay_recycler_mode.setAdapter(iplayModeAdapter);
        IplayModeAdapter iplayModeAdapter2 = new IplayModeAdapter(this.baseActivity, this.mVibrationModes);
        this.mVibrationAdapter = iplayModeAdapter2;
        this.iplay_recycler_vibration.setAdapter(iplayModeAdapter2);
        this.iplay_txt_speed.setText(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed() + "");
        this.iplay_progress_speed.setProgress(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getSpeed());
        this.iplay_txt_vibrate.setText(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibration() + "");
        this.iplay_progress_vibrate.setProgress(BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibration());
        if (MyApplication.getApplication().getmProductModel() != null) {
            PlayView_Four playView_Four = this.iplay_playView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
            PlayView_Four playView_Four2 = this.iplay_playView_vibrate;
            if (playView_Four2 != null) {
                playView_Four2.setmScreenSpeed(MyApplication.getApplication().getmProductModel().getSlide());
            }
        }
        try {
            if (ProductUtil.hasMode(0)) {
                this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
                int mode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getMode();
                if (mode > 0) {
                    this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
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
            if (ProductUtil.hasMode(1)) {
                this.mVibrationModes.addAll(Arrays.asList(ProductUtil.getProductMode(1)));
                int vibrationMode = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getVibrationMode();
                ImageView imageView = this.iplay_img_vibrate;
                if (vibrationMode >= 1) {
                    imageView.setImageResource(R.mipmap.ic_iplay_vibration_select);
                    IplayModeAdapter iplayModeAdapter4 = this.mVibrationAdapter;
                    if (iplayModeAdapter4 != null) {
                        iplayModeAdapter4.setClickPosition(vibrationMode - 1);
                    }
                } else {
                    imageView.setImageResource(R.mipmap.ic_iplay_vibration);
                }
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        this.mSpeedSafeMax = RemoteDataManager.getInstance().getLimitMax();
        this.mVibrateSafeMax = RemoteDataManager.getInstance().getLimitVibrateMax();
        scrollModeHideImg();
        scrollVibrateHideImg();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.iplay_img_mode) {
            if (this.inAnim) {
                return;
            }
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
                return;
            }
            this.isShowMode = 1;
            int visibility = this.iplay_layout_vibration.getVisibility();
            FrameLayout frameLayout = this.iplay_layout_mode;
            if (visibility == 0) {
                startAnimation(frameLayout, this.iplay_layout_vibration);
                return;
            } else {
                startAnimation(frameLayout, null);
                return;
            }
        }
        if (id == R.id.iplay_img_vibrate) {
            if (this.inAnim) {
                return;
            }
            if (this.iplay_layout_vibration.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_vibration);
                return;
            }
            this.isShowMode = 2;
            int visibility2 = this.iplay_layout_mode.getVisibility();
            FrameLayout frameLayout2 = this.iplay_layout_vibration;
            if (visibility2 == 0) {
                startAnimation(frameLayout2, this.iplay_layout_mode);
                return;
            } else {
                startAnimation(frameLayout2, null);
                return;
            }
        }
        if (id == R.id.iplay_layout_mode || id == R.id.iplay_layout_vibration) {
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
            }
            if (this.iplay_layout_vibration.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_vibration);
                return;
            }
            return;
        }
        if (id == R.id.iplay_layout_arrow_up) {
            if (this.iplay_recycler_mode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.iplay_recycler_mode.smoothScrollToPosition(0);
            return;
        }
        if (id == R.id.iplay_layout_arrow_down) {
            if (this.iplay_recycler_mode == null || this.mAdapter.getItemCount() <= 1) {
                return;
            }
            this.iplay_recycler_mode.smoothScrollToPosition(this.mAdapter.getItemCount() - 1);
            return;
        }
        if (id == R.id.iplay_layout_vibrate_arrow_up) {
            if (this.iplay_recycler_vibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
                return;
            }
            this.iplay_recycler_vibration.smoothScrollToPosition(0);
            return;
        }
        if (id != R.id.iplay_layout_vibrate_arrow_down || this.iplay_recycler_vibration == null || this.mVibrationAdapter.getItemCount() <= 1) {
            return;
        }
        this.iplay_recycler_vibration.smoothScrollToPosition(this.mVibrationAdapter.getItemCount() - 1);
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_remote_cc_smv, (ViewGroup) null);
        this.iplay_layout_bg = (ImageView) inflate.findViewById(R.id.iplay_layout_bg);
        this.iplay_playView = (PlayView_Four) inflate.findViewById(R.id.iplay_playView);
        this.iplay_playView_vibrate = (PlayView_Four) inflate.findViewById(R.id.iplay_playView_vibrate);
        this.iplay_progress_speed = (VerticalProgressBar) inflate.findViewById(R.id.iplay_progress_speed);
        this.iplay_progress_vibrate = (VerticalProgressBar) inflate.findViewById(R.id.iplay_progress_vibrate);
        this.iplay_txt_speed = (TextView) inflate.findViewById(R.id.iplay_txt_speed);
        this.iplay_txt_vibrate = (TextView) inflate.findViewById(R.id.iplay_txt_vibrate);
        this.iplay_img_mode = (ImageView) inflate.findViewById(R.id.iplay_img_mode);
        this.iplay_img_vibrate = (ImageView) inflate.findViewById(R.id.iplay_img_vibrate);
        this.iplay_layout_mode = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode);
        this.iplay_layout_vibration = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibration);
        this.iplay_layout_mode_border = (LinearLayout) inflate.findViewById(R.id.iplay_layout_mode_border);
        this.iplay_layout_vibration_border = (LinearLayout) inflate.findViewById(R.id.iplay_layout_vibration_border);
        this.iplay_recycler_mode = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_mode);
        this.iplay_recycler_vibration = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_vibration);
        this.iplay_layout_arrow_up = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_down);
        this.iplay_layout_vibrate_arrow_up = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.iplay_layout_vibrate_arrow_down = (FrameLayout) inflate.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.iplay_layout_speed_safe = (ViewGroup) inflate.findViewById(R.id.iplay_layout_speed_safe);
        this.iplay_layout_vibrate_safe = (ViewGroup) inflate.findViewById(R.id.iplay_layout_vibrate_safe);
        this.iplay_txt_speed_safe = (TextView) inflate.findViewById(R.id.iplay_txt_speed_safe);
        this.iplay_txt_vibrate_safe = (TextView) inflate.findViewById(R.id.iplay_txt_vibrate_safe);
        this.iplay_playView.setProgressBar(this.iplay_progress_speed);
        this.iplay_playView.setTouch(false);
        this.iplay_playView.setMy(false);
        this.iplay_playView_vibrate.setProgressBar(this.iplay_progress_vibrate);
        this.iplay_playView_vibrate.setTouch(false);
        this.iplay_playView.setMy(false);
        return inflate;
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshAutoUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshLinePoints(Map<String, Point> map) {
        PlayView_Four playView_Four;
        if (map == null || (playView_Four = this.iplay_playView) == null) {
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
    public void refreshSmoothnessUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshSpeedModeUI(int i) {
        try {
            if (i <= 0) {
                this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
            } else {
                this.isShowMode = 1;
                if (this.iplay_layout_mode.getVisibility() != 0) {
                    startAnimation(this.iplay_layout_mode, this.iplay_layout_vibration);
                }
                this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
            }
            IplayModeAdapter iplayModeAdapter = this.mAdapter;
            if (iplayModeAdapter != null) {
                int i2 = i - 1;
                iplayModeAdapter.setClickPosition(i2);
                if (i > 0) {
                    this.iplay_recycler_mode.smoothScrollToPosition(i2);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshSpeedUI(int i) {
        try {
            this.iplay_txt_speed.setText(i + "");
            this.iplay_progress_speed.setProgress(i);
            this.isShowMode = 0;
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
            }
            this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
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
        PlayView_Four playView_Four;
        if (map == null || (playView_Four = this.iplay_playView_vibrate) == null) {
            return;
        }
        playView_Four.setPointArr(map);
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshVibrateModeUI(int i) {
        try {
            if (i >= 1) {
                this.isShowMode = 2;
                if (this.iplay_layout_vibration.getVisibility() != 0) {
                    startAnimation(this.iplay_layout_vibration, this.iplay_layout_mode);
                }
                this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration_select);
            } else {
                this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
            }
            IplayModeAdapter iplayModeAdapter = this.mVibrationAdapter;
            if (iplayModeAdapter != null) {
                int i2 = i - 1;
                iplayModeAdapter.setClickPosition(i2);
                if (i > 0) {
                    this.iplay_recycler_vibration.smoothScrollToPosition(i2);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshVibrateUI(int i) {
        try {
            this.iplay_txt_vibrate.setText(i + "");
            this.iplay_progress_vibrate.setProgress(i);
            this.isShowMode = 0;
            if (this.iplay_layout_vibration.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_vibration);
            }
            this.iplay_img_vibrate.setImageResource(R.mipmap.ic_iplay_vibration);
            IplayModeAdapter iplayModeAdapter = this.mVibrationAdapter;
            if (iplayModeAdapter != null) {
                iplayModeAdapter.setClickPosition(-1);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshWebSpeedUI(int i) {
        VerticalProgressBar verticalProgressBar = this.iplay_progress_speed;
        if (verticalProgressBar != null) {
            verticalProgressBar.setSecondaryProgress(i);
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshWebVibrateUI(int i) {
        VerticalProgressBar verticalProgressBar = this.iplay_progress_vibrate;
        if (verticalProgressBar != null) {
            verticalProgressBar.setSecondaryProgress(i);
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void setListener() {
        super.setListener();
        this.iplay_layout_speed_safe.setOnTouchListener(this.speedSafeOnTouch);
        this.iplay_layout_vibrate_safe.setOnTouchListener(this.vibrateSafeOnTouch);
        this.iplay_img_mode.setOnClickListener(this);
        this.iplay_img_vibrate.setOnClickListener(this);
        this.iplay_layout_mode.setOnClickListener(this);
        this.iplay_layout_vibration.setOnClickListener(this);
        this.iplay_layout_arrow_up.setOnClickListener(this);
        this.iplay_layout_arrow_down.setOnClickListener(this);
        this.iplay_layout_vibrate_arrow_up.setOnClickListener(this);
        this.iplay_layout_vibrate_arrow_down.setOnClickListener(this);
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.1
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                RemoteCC_SMVPlayFragment.this.scrollModeHideImg();
            }
        });
        this.iplay_recycler_vibration.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.2
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                RemoteCC_SMVPlayFragment.this.scrollVibrateHideImg();
            }
        });
        this.iplay_layout_speed_safe.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.3
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                RemoteCC_SMVPlayFragment.this.iplay_layout_speed_safe.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                RemoteCC_SMVPlayFragment remoteCC_SMVPlayFragment = RemoteCC_SMVPlayFragment.this;
                remoteCC_SMVPlayFragment.refreshSafeUI(remoteCC_SMVPlayFragment.mSpeedSafeMax);
            }
        });
        this.iplay_layout_vibrate_safe.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment.4
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                RemoteCC_SMVPlayFragment.this.iplay_layout_vibrate_safe.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                RemoteCC_SMVPlayFragment remoteCC_SMVPlayFragment = RemoteCC_SMVPlayFragment.this;
                remoteCC_SMVPlayFragment.refreshVibrateSafeUI(remoteCC_SMVPlayFragment.mVibrateSafeMax);
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void startHeart() {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void stopHeart() {
    }
}
