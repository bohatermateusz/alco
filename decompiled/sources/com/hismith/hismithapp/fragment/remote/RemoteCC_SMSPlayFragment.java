package com.hismith.hismithapp.fragment.remote;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.graphics.Point;
import android.os.Bundle;
import android.os.Handler;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.animation.LinearInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.IplayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RemoteDataManager;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.PlayView_Four;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

/* loaded from: classes3.dex */
public class RemoteCC_SMSPlayFragment extends RemoteControlFragment implements View.OnClickListener {
    private float downX;
    private float endDownX;
    private ImageView img_run_mode;
    private ImageView iplay_img_mic;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_position;
    private RelativeLayout iplay_layout;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
    private ImageView iplay_layout_bg;
    private FrameLayout iplay_layout_mic_btn;
    private FrameLayout iplay_layout_mode;
    private FrameLayout iplay_layout_mode_btn;
    private FrameLayout iplay_layout_touch_position;
    private PlayView_Four iplay_playView;
    private RecyclerView iplay_recycler_mode;
    private ConstraintLayout layout_position_max;
    private ConstraintLayout layout_position_min;
    private FrameLayout layout_speed_max;
    private IplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private GridLayoutManager mLayoutManager;
    private ProgressBar progress_position;
    private ProgressBar progress_speed;
    private CustomSeekbar seek_smooth;
    private float startDownX;
    private TextView txt_position;
    private TextView txt_position_label;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private TextView txt_smooth;
    private TextView txt_smooth_label;
    private TextView txt_speed;
    private TextView txt_speed_label;
    private TextView txt_speed_max;
    private int start_stop_position_diff = Constans.start_stop_position_diff;
    private int delay = 15;
    private Handler mHandler = new Handler();
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;
    private int mSafeMax = 100;
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.8
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                RemoteCC_SMSPlayFragment.this.downX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - RemoteCC_SMSPlayFragment.this.downX;
            int right = RemoteCC_SMSPlayFragment.this.progress_speed.getRight() + (RemoteCC_SMSPlayFragment.this.layout_speed_max.getWidth() / 2);
            LogUtil.e("getRight():" + RemoteCC_SMSPlayFragment.this.layout_speed_max.getRight() + ", width:" + RemoteCC_SMSPlayFragment.this.progress_speed.getRight() + ", " + RemoteCC_SMSPlayFragment.this.layout_speed_max.getWidth());
            int right2 = (int) (right - (RemoteCC_SMSPlayFragment.this.layout_speed_max.getRight() + x));
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > RemoteCC_SMSPlayFragment.this.progress_speed.getWidth()) {
                right2 = RemoteCC_SMSPlayFragment.this.progress_speed.getWidth();
            }
            LogUtil.e("safe right:" + right2);
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) RemoteCC_SMSPlayFragment.this.layout_speed_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            RemoteCC_SMSPlayFragment.this.layout_speed_max.setLayoutParams(layoutParams);
            int width = (int) (100.0d - ((right2 * 100.0d) / RemoteCC_SMSPlayFragment.this.progress_speed.getWidth()));
            RemoteCC_SMSPlayFragment.this.mSafeMax = width;
            RemoteCC_SMSPlayFragment.this.txt_speed_max.setText(width + "");
            RemoteDataManager.getInstance().setLimitMax(RemoteCC_SMSPlayFragment.this.mSafeMax);
            if (BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).getmConnBleDevice() != null) {
                BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).getmConnBleDevice().getSpeed();
            }
            if (width >= 0 && width < RemoteCC_SMSPlayFragment.this.progress_speed.getSecondaryProgress()) {
                BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).writeSpeed(width);
                RemoteCC_SMSPlayFragment.this.progress_speed.setProgress(width);
                RemoteCC_SMSPlayFragment.this.txt_speed.setText(width + "");
                return true;
            }
            if (width < RemoteCC_SMSPlayFragment.this.progress_speed.getSecondaryProgress() || RemoteCC_SMSPlayFragment.this.progress_speed.getProgress() == RemoteCC_SMSPlayFragment.this.progress_speed.getSecondaryProgress()) {
                return true;
            }
            int secondaryProgress = RemoteCC_SMSPlayFragment.this.progress_speed.getSecondaryProgress();
            BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).writeSpeed(secondaryProgress);
            RemoteCC_SMSPlayFragment.this.progress_speed.setProgress(secondaryProgress);
            RemoteCC_SMSPlayFragment.this.txt_speed.setText(secondaryProgress + "");
            return true;
        }
    };
    private int mPositionEnd = 100;
    private View.OnTouchListener endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.9
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                RemoteCC_SMSPlayFragment.this.endDownX = motionEvent.getX();
            } else if (action == 2) {
                float x = motionEvent.getX() - RemoteCC_SMSPlayFragment.this.endDownX;
                int width = RemoteCC_SMSPlayFragment.this.progress_position.getWidth();
                int right = (int) ((RemoteCC_SMSPlayFragment.this.progress_position.getRight() + (RemoteCC_SMSPlayFragment.this.layout_position_max.getWidth() / 2)) - (RemoteCC_SMSPlayFragment.this.layout_position_max.getRight() + x));
                LogUtil.e("onTouch moveX:" + x + ", right():" + right);
                if (right < 0) {
                    right = 0;
                } else if (right > width) {
                    right = width;
                }
                double d = width;
                int i = (int) (100.0d - ((right * 100.0d) / d));
                if (i <= RemoteCC_SMSPlayFragment.this.mPositionStart + RemoteCC_SMSPlayFragment.this.start_stop_position_diff) {
                    int i2 = RemoteCC_SMSPlayFragment.this.mPositionStart + RemoteCC_SMSPlayFragment.this.start_stop_position_diff;
                    i = i2 > 100 ? 100 : i2;
                    right = (int) (d - ((d / 100.0d) * i));
                }
                int i3 = i <= 100 ? i < 0 ? 0 : i : 100;
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) RemoteCC_SMSPlayFragment.this.layout_position_max.getLayoutParams();
                layoutParams.setMargins(0, 0, right, 0);
                RemoteCC_SMSPlayFragment.this.layout_position_max.setLayoutParams(layoutParams);
                if (RemoteCC_SMSPlayFragment.this.mPositionEnd != i3) {
                    RemoteCC_SMSPlayFragment.this.mPositionEnd = i3;
                    RemoteDataManager.getInstance().setLimitStopPosition(RemoteCC_SMSPlayFragment.this.mPositionEnd);
                    LogUtil.e("结束位置：" + RemoteCC_SMSPlayFragment.this.mPositionEnd + ", right:" + right);
                    RemoteCC_SMSPlayFragment.this.txt_position_max.setText(i3 + "");
                    SocketManager.getInstance().setStroke(RemoteCC_SMSPlayFragment.this.mPositionStart, RemoteCC_SMSPlayFragment.this.mPositionEnd, 1);
                }
            }
            return true;
        }
    };
    private int mPositionStart = 0;
    private View.OnTouchListener startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.10
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                RemoteCC_SMSPlayFragment.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
            } else if (action == 2) {
                float x = motionEvent.getX() - RemoteCC_SMSPlayFragment.this.startDownX;
                LogUtil.e("抖动：" + motionEvent.getX() + ", " + RemoteCC_SMSPlayFragment.this.startDownX + " = " + x);
                int width = RemoteCC_SMSPlayFragment.this.progress_position.getWidth();
                int left = (int) ((RemoteCC_SMSPlayFragment.this.layout_position_min.getLeft() + x) - (RemoteCC_SMSPlayFragment.this.progress_position.getLeft() - (RemoteCC_SMSPlayFragment.this.layout_position_min.getWidth() / 2)));
                LogUtil.e("onTouch moveX:" + x + ", left():" + left);
                if (left < 0) {
                    left = 0;
                } else if (left > width) {
                    left = width;
                }
                double d = width;
                int i = (int) ((left * 100.0d) / d);
                if (i >= RemoteCC_SMSPlayFragment.this.mPositionEnd - RemoteCC_SMSPlayFragment.this.start_stop_position_diff) {
                    int i2 = RemoteCC_SMSPlayFragment.this.mPositionEnd - RemoteCC_SMSPlayFragment.this.start_stop_position_diff;
                    i = i2 < 0 ? 0 : i2;
                    left = (int) ((d / 100.0d) * i);
                }
                if (i > 100) {
                    i = 100;
                } else if (i < 0) {
                    i = 0;
                }
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) RemoteCC_SMSPlayFragment.this.layout_position_min.getLayoutParams();
                layoutParams.setMargins(left, 0, 0, 0);
                RemoteCC_SMSPlayFragment.this.layout_position_min.setLayoutParams(layoutParams);
                if (RemoteCC_SMSPlayFragment.this.mPositionStart != i) {
                    RemoteCC_SMSPlayFragment.this.mPositionStart = i;
                    RemoteDataManager.getInstance().setLimitStartPosition(RemoteCC_SMSPlayFragment.this.mPositionStart);
                    LogUtil.e("开始位置：" + RemoteCC_SMSPlayFragment.this.mPositionStart + ", left:" + left);
                    RemoteCC_SMSPlayFragment.this.txt_position_min.setText(i + "");
                    SocketManager.getInstance().setStroke(RemoteCC_SMSPlayFragment.this.mPositionStart, RemoteCC_SMSPlayFragment.this.mPositionEnd, 1);
                }
            }
            return true;
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleDataUI() {
        LogUtil.e("initBleDataUI()");
        if (BleManager.getInstance(this.baseActivity).getmConnBleDevice() != null) {
            int startPosition = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStartPosition();
            onStartPositionUI(startPosition);
            int stopPosition = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStopPosition();
            int i = stopPosition - startPosition;
            int i2 = this.start_stop_position_diff;
            if (i < i2) {
                int i3 = startPosition + i2;
                if (i3 > 100) {
                    i3 = 100;
                }
                BleManager.getInstance(this.baseActivity).writeStopPosition(i3);
                onStopPositionUI(stopPosition);
            } else {
                if (stopPosition <= 0) {
                    stopPosition = startPosition + i2;
                }
                onStopPositionUI(stopPosition);
            }
            refreshSafeUI(this.mSafeMax);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initModeShow() {
        int i = this.isShowMode;
        if (i != 0) {
            if (i == 1) {
                startAnimation(this.iplay_layout_mode);
            }
        } else {
            FrameLayout frameLayout = this.iplay_layout_mode;
            if (frameLayout == null || frameLayout.getVisibility() != 0) {
                return;
            }
            stopAnimation(this.iplay_layout_mode);
        }
    }

    private void initProductData() {
        ProductModel productModel = MyApplication.getApplication().getmProductModel();
        LogUtil.e("BleManager 初始化：" + productModel);
        if (productModel == null) {
            return;
        }
        PlayView_Four playView_Four = this.iplay_playView;
        if (playView_Four != null) {
            playView_Four.setmScreenSpeed(productModel.getSlide());
        }
        this.mModes.clear();
        boolean hasMode = ProductUtil.hasMode(0);
        FrameLayout frameLayout = this.iplay_layout_mode_btn;
        if (hasMode) {
            frameLayout.setVisibility(0);
            this.mModes.addAll(Arrays.asList(ProductUtil.getProductMode(0)));
        } else {
            frameLayout.setVisibility(8);
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
    }

    private void onStartPositionUI(int i) {
        try {
            if (this.mPositionStart != i) {
                this.mPositionStart = i;
                RemoteDataManager.getInstance().setLimitStartPosition(i);
                this.txt_position_min.setText(i + "");
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_min.getLayoutParams();
                layoutParams.setMargins((int) ((this.progress_position.getWidth() / 100.0d) * i), 0, 0, 0);
                this.layout_position_min.setLayoutParams(layoutParams);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void onStopPositionUI(int i) {
        try {
            if (this.mPositionEnd != i) {
                this.mPositionEnd = i;
                RemoteDataManager.getInstance().setLimitStopPosition(i);
                this.txt_position_max.setText(i + "");
                double width = this.progress_position.getWidth();
                int i2 = (int) (width - ((width / 100.0d) * i));
                ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_max.getLayoutParams();
                layoutParams.setMargins(0, 0, i2, 0);
                this.layout_position_max.setLayoutParams(layoutParams);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void refreshSafeUI(int i) {
        this.txt_speed_max.setText(i + "");
        double width = this.progress_speed.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_speed_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_speed_max.setLayoutParams(layoutParams);
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

    private void startAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 0.0f, 1.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.6
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteCC_SMSPlayFragment.this.inAnim = false;
                if (i != RemoteCC_SMSPlayFragment.this.isShowMode) {
                    RemoteCC_SMSPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteCC_SMSPlayFragment.this.inAnim = true;
                frameLayout.setVisibility(0);
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
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
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.7
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                RemoteCC_SMSPlayFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != RemoteCC_SMSPlayFragment.this.isShowMode) {
                    RemoteCC_SMSPlayFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                RemoteCC_SMSPlayFragment.this.inAnim = true;
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
        this.mModes = new ArrayList();
        IplayModeAdapter iplayModeAdapter = new IplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = iplayModeAdapter;
        this.iplay_recycler_mode.setAdapter(iplayModeAdapter);
        this.mSafeMax = RemoteDataManager.getInstance().getLimitMax();
        initProductData();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.img_run_mode) {
            if (BleManager.getInstance(this.baseActivity).getmConnBleDevice() == null || BleManager.getInstance(this.baseActivity).getmConnBleDevice().getRunMode() != 0) {
                BleManager.getInstance(this.baseActivity).writeRunAuto();
                return;
            } else {
                BleManager.getInstance(this.baseActivity).writeRunManual();
                return;
            }
        }
        if (view.getId() != R.id.iplay_layout_mode_btn || this.inAnim) {
            return;
        }
        if (this.iplay_layout_mode.getVisibility() == 0) {
            stopAnimation(this.iplay_layout_mode);
        } else {
            this.isShowMode = 1;
            startAnimation(this.iplay_layout_mode);
        }
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_remote_cc_sms, (ViewGroup) null);
        this.iplay_layout = (RelativeLayout) inflate.findViewById(R.id.iplay_layout);
        this.iplay_layout_bg = (ImageView) inflate.findViewById(R.id.iplay_layout_bg);
        this.img_run_mode = (ImageView) inflate.findViewById(R.id.img_run_mode);
        this.iplay_layout_mic_btn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mic_btn);
        this.iplay_img_mic = (ImageView) inflate.findViewById(R.id.iplay_img_mic);
        this.iplay_layout_mode_btn = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode_btn);
        this.iplay_img_mode = (ImageView) inflate.findViewById(R.id.iplay_img_mode);
        this.txt_speed_label = (TextView) inflate.findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) inflate.findViewById(R.id.txt_speed);
        this.progress_speed = (ProgressBar) inflate.findViewById(R.id.progress_speed);
        this.layout_speed_max = (FrameLayout) inflate.findViewById(R.id.layout_speed_max);
        this.txt_speed_max = (TextView) inflate.findViewById(R.id.txt_speed_max);
        this.txt_position_label = (TextView) inflate.findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) inflate.findViewById(R.id.txt_position);
        this.progress_position = (ProgressBar) inflate.findViewById(R.id.progress_position);
        this.layout_position_max = (ConstraintLayout) inflate.findViewById(R.id.layout_position_max);
        this.txt_position_max = (TextView) inflate.findViewById(R.id.txt_position_max);
        this.layout_position_min = (ConstraintLayout) inflate.findViewById(R.id.layout_position_min);
        this.txt_position_min = (TextView) inflate.findViewById(R.id.txt_position_min);
        this.txt_smooth_label = (TextView) inflate.findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) inflate.findViewById(R.id.txt_smooth);
        this.seek_smooth = (CustomSeekbar) inflate.findViewById(R.id.seek_smooth);
        this.iplay_playView = (PlayView_Four) inflate.findViewById(R.id.iplay_playView);
        this.iplay_layout_touch_position = (FrameLayout) inflate.findViewById(R.id.iplay_layout_touch_position);
        this.iplay_img_position = (ImageView) inflate.findViewById(R.id.iplay_img_position);
        this.iplay_layout_mode = (FrameLayout) inflate.findViewById(R.id.iplay_layout_mode);
        this.iplay_recycler_mode = (RecyclerView) inflate.findViewById(R.id.iplay_recycler_mode);
        this.iplay_layout_arrow_up = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) inflate.findViewById(R.id.iplay_layout_arrow_down);
        LanguageUtil.setText(this.baseActivity, this.txt_speed_label, "speedTitle");
        this.txt_position_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "position") + ":");
        this.txt_smooth_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "smoothness") + ":");
        this.iplay_playView.setProgressBar(this.progress_speed);
        this.iplay_playView.setTouch(false);
        this.iplay_playView.setMy(false);
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.3
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + RemoteCC_SMSPlayFragment.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + RemoteCC_SMSPlayFragment.this.seek_smooth.getProgress());
                int progress = RemoteCC_SMSPlayFragment.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                int i3 = i / 100;
                RemoteCC_SMSPlayFragment.this.txt_smooth.setText(i3 + "");
                BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).writeSmooth(i3);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.3.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        RemoteCC_SMSPlayFragment.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
        return inflate;
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment, androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        if (ProductUtil.isSMSType()) {
            int stopPosition = BleManager.getInstance(this.baseActivity).getmConnBleDevice().getStopPosition();
            int i = this.delay;
            int i2 = i * 2;
            if (this.mPositionStart >= stopPosition) {
                i2 = i;
                i *= 2;
            }
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.1
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).writeData(BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).getBleCCData(7, RemoteCC_SMSPlayFragment.this.mPositionStart));
                }
            }, i);
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.2
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).writeData(BleManager.getInstance(RemoteCC_SMSPlayFragment.this.baseActivity).getBleCCData(8, RemoteCC_SMSPlayFragment.this.mPositionEnd));
                }
            }, i2);
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshAutoUI(int i) {
        ImageView imageView;
        if (i == 0) {
            ImageView imageView2 = this.img_run_mode;
            if (imageView2 != null) {
                imageView2.setImageResource(R.mipmap.ic_auto_switch_on);
                return;
            }
            return;
        }
        if (i != 1 || (imageView = this.img_run_mode) == null) {
            return;
        }
        imageView.setImageResource(R.mipmap.ic_auto_switch_off);
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
        TextView textView = this.txt_position;
        if (textView != null) {
            textView.setText(i + "");
        }
        ProgressBar progressBar = this.progress_position;
        if (progressBar != null) {
            progressBar.setProgress(i);
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshSmoothnessUI(int i) {
        try {
            this.txt_smooth.setText(i + "");
            ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
            ofInt.setInterpolator(new LinearInterpolator());
            ofInt.setDuration(300L);
            ofInt.setRepeatCount(0);
            ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.11
                @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                public void onAnimationUpdate(ValueAnimator valueAnimator) {
                    LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                    RemoteCC_SMSPlayFragment.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                }
            });
            ofInt.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshSpeedModeUI(int i) {
        ImageView imageView = this.iplay_img_mode;
        try {
            if (i <= 0) {
                imageView.setImageResource(R.mipmap.ic_iplay_mode);
                this.isShowMode = 0;
                if (this.iplay_layout_mode.getVisibility() == 0) {
                    stopAnimation(this.iplay_layout_mode);
                }
            } else {
                imageView.setImageResource(R.mipmap.ic_iplay_mode_select);
                this.isShowMode = 1;
                if (this.iplay_layout_mode.getVisibility() != 0) {
                    startAnimation(this.iplay_layout_mode);
                }
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
        ProgressBar progressBar = this.progress_speed;
        if (progressBar != null) {
            progressBar.setProgress(i);
        }
        TextView textView = this.txt_speed;
        if (textView != null) {
            textView.setText(i + "");
        }
        refreshSpeedModeUI(0);
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
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshVibrateUI(int i) {
    }

    @Override // com.hismith.hismithapp.fragment.remote.RemoteControlFragment
    public void refreshWebSpeedUI(int i) {
        ProgressBar progressBar = this.progress_speed;
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
        this.progress_position.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.4
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                RemoteCC_SMSPlayFragment.this.progress_position.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                LogUtil.e("onGlobalLayout " + RemoteCC_SMSPlayFragment.this.progress_position.getWidth());
                RemoteCC_SMSPlayFragment.this.initBleDataUI();
            }
        });
        this.iplay_layout_mode_btn.setOnClickListener(this);
        this.img_run_mode.setOnClickListener(this);
        this.layout_speed_max.setOnTouchListener(this.safeOnTouch);
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment.5
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                RemoteCC_SMSPlayFragment.this.scrollModeHideImg();
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
