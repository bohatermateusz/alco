package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.PlayView_Four;

/* loaded from: classes3.dex */
public final class FragmentRemoteCcSmsBinding implements ViewBinding {
    public final View guideView;
    public final ImageView imgRunMode;
    public final ImageView iplayImgMode;
    public final ImageView iplayImgPosition;
    public final RelativeLayout iplayLayout;
    public final FrameLayout iplayLayoutArrowDown;
    public final FrameLayout iplayLayoutArrowUp;
    public final ImageView iplayLayoutBg;
    public final FrameLayout iplayLayoutMode;
    public final LinearLayout iplayLayoutModeBorder;
    public final FrameLayout iplayLayoutModeBtn;
    public final FrameLayout iplayLayoutTouchPosition;
    public final PlayView_Four iplayPlayView;
    public final RecyclerView iplayRecyclerMode;
    public final ConstraintLayout layoutPositionMax;
    public final ConstraintLayout layoutPositionMin;
    public final FrameLayout layoutSpeedMax;
    public final Guideline lineMaxStart;
    public final Guideline lineMinStart;
    public final ProgressBar progressPosition;
    public final ProgressBar progressSpeed;
    private final LinearLayout rootView;
    public final CustomSeekbar seekSmooth;
    public final TextView txtPosition;
    public final TextView txtPositionLabel;
    public final AppCompatTextView txtPositionMax;
    public final AppCompatTextView txtPositionMin;
    public final TextView txtSmooth;
    public final TextView txtSmoothLabel;
    public final TextView txtSpeed;
    public final TextView txtSpeedLabel;
    public final AppCompatTextView txtSpeedMax;

    private FragmentRemoteCcSmsBinding(LinearLayout linearLayout, View view, ImageView imageView, ImageView imageView2, ImageView imageView3, RelativeLayout relativeLayout, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView4, FrameLayout frameLayout3, LinearLayout linearLayout2, FrameLayout frameLayout4, FrameLayout frameLayout5, PlayView_Four playView_Four, RecyclerView recyclerView, ConstraintLayout constraintLayout, ConstraintLayout constraintLayout2, FrameLayout frameLayout6, Guideline guideline, Guideline guideline2, ProgressBar progressBar, ProgressBar progressBar2, CustomSeekbar customSeekbar, TextView textView, TextView textView2, AppCompatTextView appCompatTextView, AppCompatTextView appCompatTextView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, AppCompatTextView appCompatTextView3) {
        this.rootView = linearLayout;
        this.guideView = view;
        this.imgRunMode = imageView;
        this.iplayImgMode = imageView2;
        this.iplayImgPosition = imageView3;
        this.iplayLayout = relativeLayout;
        this.iplayLayoutArrowDown = frameLayout;
        this.iplayLayoutArrowUp = frameLayout2;
        this.iplayLayoutBg = imageView4;
        this.iplayLayoutMode = frameLayout3;
        this.iplayLayoutModeBorder = linearLayout2;
        this.iplayLayoutModeBtn = frameLayout4;
        this.iplayLayoutTouchPosition = frameLayout5;
        this.iplayPlayView = playView_Four;
        this.iplayRecyclerMode = recyclerView;
        this.layoutPositionMax = constraintLayout;
        this.layoutPositionMin = constraintLayout2;
        this.layoutSpeedMax = frameLayout6;
        this.lineMaxStart = guideline;
        this.lineMinStart = guideline2;
        this.progressPosition = progressBar;
        this.progressSpeed = progressBar2;
        this.seekSmooth = customSeekbar;
        this.txtPosition = textView;
        this.txtPositionLabel = textView2;
        this.txtPositionMax = appCompatTextView;
        this.txtPositionMin = appCompatTextView2;
        this.txtSmooth = textView3;
        this.txtSmoothLabel = textView4;
        this.txtSpeed = textView5;
        this.txtSpeedLabel = textView6;
        this.txtSpeedMax = appCompatTextView3;
    }

    public static FragmentRemoteCcSmsBinding bind(View view) {
        int i = R.id.guide_view;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById != null) {
            i = R.id.img_run_mode;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.iplay_img_mode;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.iplay_img_position;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.iplay_layout;
                        RelativeLayout relativeLayout = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                        if (relativeLayout != null) {
                            i = R.id.iplay_layout_arrow_down;
                            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                            if (frameLayout != null) {
                                i = R.id.iplay_layout_arrow_up;
                                FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                if (frameLayout2 != null) {
                                    i = R.id.iplay_layout_bg;
                                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView4 != null) {
                                        i = R.id.iplay_layout_mode;
                                        FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                        if (frameLayout3 != null) {
                                            i = R.id.iplay_layout_mode_border;
                                            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout != null) {
                                                i = R.id.iplay_layout_mode_btn;
                                                FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                if (frameLayout4 != null) {
                                                    i = R.id.iplay_layout_touch_position;
                                                    FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                    if (frameLayout5 != null) {
                                                        i = R.id.iplay_playView;
                                                        PlayView_Four playView_Four = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                                        if (playView_Four != null) {
                                                            i = R.id.iplay_recycler_mode;
                                                            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                            if (recyclerView != null) {
                                                                i = R.id.layout_position_max;
                                                                ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                if (constraintLayout != null) {
                                                                    i = R.id.layout_position_min;
                                                                    ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (constraintLayout2 != null) {
                                                                        i = R.id.layout_speed_max;
                                                                        FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (frameLayout6 != null) {
                                                                            i = R.id.line_max_start;
                                                                            Guideline guideline = (Guideline) ViewBindings.findChildViewById(view, i);
                                                                            if (guideline != null) {
                                                                                i = R.id.line_min_start;
                                                                                Guideline guideline2 = (Guideline) ViewBindings.findChildViewById(view, i);
                                                                                if (guideline2 != null) {
                                                                                    i = R.id.progress_position;
                                                                                    ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                    if (progressBar != null) {
                                                                                        i = R.id.progress_speed;
                                                                                        ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                        if (progressBar2 != null) {
                                                                                            i = R.id.seek_smooth;
                                                                                            CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
                                                                                            if (customSeekbar != null) {
                                                                                                i = R.id.txt_position;
                                                                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView != null) {
                                                                                                    i = R.id.txt_position_label;
                                                                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView2 != null) {
                                                                                                        i = R.id.txt_position_max;
                                                                                                        AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (appCompatTextView != null) {
                                                                                                            i = R.id.txt_position_min;
                                                                                                            AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (appCompatTextView2 != null) {
                                                                                                                i = R.id.txt_smooth;
                                                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView3 != null) {
                                                                                                                    i = R.id.txt_smooth_label;
                                                                                                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (textView4 != null) {
                                                                                                                        i = R.id.txt_speed;
                                                                                                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (textView5 != null) {
                                                                                                                            i = R.id.txt_speed_label;
                                                                                                                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (textView6 != null) {
                                                                                                                                i = R.id.txt_speed_max;
                                                                                                                                AppCompatTextView appCompatTextView3 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (appCompatTextView3 != null) {
                                                                                                                                    return new FragmentRemoteCcSmsBinding((LinearLayout) view, findChildViewById, imageView, imageView2, imageView3, relativeLayout, frameLayout, frameLayout2, imageView4, frameLayout3, linearLayout, frameLayout4, frameLayout5, playView_Four, recyclerView, constraintLayout, constraintLayout2, frameLayout6, guideline, guideline2, progressBar, progressBar2, customSeekbar, textView, textView2, appCompatTextView, appCompatTextView2, textView3, textView4, textView5, textView6, appCompatTextView3);
                                                                                                                                }
                                                                                                                            }
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static FragmentRemoteCcSmsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentRemoteCcSmsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_remote_cc_sms, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public LinearLayout getRoot() {
        return this.rootView;
    }
}
