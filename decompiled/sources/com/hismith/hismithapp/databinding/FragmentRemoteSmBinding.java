package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public final class FragmentRemoteSmBinding implements ViewBinding {
    public final ImageView imgReversal;
    public final ImageView iplayImgHeart;
    public final ImageView iplayImgMode;
    public final ImageView iplayImgVibrate;
    public final FrameLayout iplayLayoutArrowDown;
    public final FrameLayout iplayLayoutArrowUp;
    public final ImageView iplayLayoutBg;
    public final FrameLayout iplayLayoutMode;
    public final LinearLayout iplayLayoutModeBorder;
    public final FrameLayout iplayLayoutModeBtn;
    public final FrameLayout iplayLayoutSpeedSafe;
    public final FrameLayout iplayLayoutVibrateArrowDown;
    public final FrameLayout iplayLayoutVibrateArrowUp;
    public final FrameLayout iplayLayoutVibrateBtn;
    public final FrameLayout iplayLayoutVibration;
    public final LinearLayout iplayLayoutVibrationBorder;
    public final PlayView_Four iplayPlayView;
    public final VerticalProgressBar iplayProgressSpeed;
    public final RecyclerView iplayRecyclerMode;
    public final RecyclerView iplayRecyclerVibration;
    public final TextView iplayTxtMinSpeed;
    public final TextView iplayTxtSpeed;
    public final TextView iplayTxtSpeedSafe;
    public final ConstraintLayout layoutAction;
    private final LinearLayout rootView;
    public final TextView titleTxtStatus;

    private FragmentRemoteSmBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView5, FrameLayout frameLayout3, LinearLayout linearLayout2, FrameLayout frameLayout4, FrameLayout frameLayout5, FrameLayout frameLayout6, FrameLayout frameLayout7, FrameLayout frameLayout8, FrameLayout frameLayout9, LinearLayout linearLayout3, PlayView_Four playView_Four, VerticalProgressBar verticalProgressBar, RecyclerView recyclerView, RecyclerView recyclerView2, TextView textView, TextView textView2, TextView textView3, ConstraintLayout constraintLayout, TextView textView4) {
        this.rootView = linearLayout;
        this.imgReversal = imageView;
        this.iplayImgHeart = imageView2;
        this.iplayImgMode = imageView3;
        this.iplayImgVibrate = imageView4;
        this.iplayLayoutArrowDown = frameLayout;
        this.iplayLayoutArrowUp = frameLayout2;
        this.iplayLayoutBg = imageView5;
        this.iplayLayoutMode = frameLayout3;
        this.iplayLayoutModeBorder = linearLayout2;
        this.iplayLayoutModeBtn = frameLayout4;
        this.iplayLayoutSpeedSafe = frameLayout5;
        this.iplayLayoutVibrateArrowDown = frameLayout6;
        this.iplayLayoutVibrateArrowUp = frameLayout7;
        this.iplayLayoutVibrateBtn = frameLayout8;
        this.iplayLayoutVibration = frameLayout9;
        this.iplayLayoutVibrationBorder = linearLayout3;
        this.iplayPlayView = playView_Four;
        this.iplayProgressSpeed = verticalProgressBar;
        this.iplayRecyclerMode = recyclerView;
        this.iplayRecyclerVibration = recyclerView2;
        this.iplayTxtMinSpeed = textView;
        this.iplayTxtSpeed = textView2;
        this.iplayTxtSpeedSafe = textView3;
        this.layoutAction = constraintLayout;
        this.titleTxtStatus = textView4;
    }

    public static FragmentRemoteSmBinding bind(View view) {
        int i = R.id.img_reversal;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.iplay_img_heart;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.iplay_img_mode;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.iplay_img_vibrate;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.iplay_layout_arrow_down;
                        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                        if (frameLayout != null) {
                            i = R.id.iplay_layout_arrow_up;
                            FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                            if (frameLayout2 != null) {
                                i = R.id.iplay_layout_bg;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.iplay_layout_mode;
                                    FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                    if (frameLayout3 != null) {
                                        i = R.id.iplay_layout_mode_border;
                                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout != null) {
                                            i = R.id.iplay_layout_mode_btn;
                                            FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                            if (frameLayout4 != null) {
                                                i = R.id.iplay_layout_speed_safe;
                                                FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                if (frameLayout5 != null) {
                                                    i = R.id.iplay_layout_vibrate_arrow_down;
                                                    FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                    if (frameLayout6 != null) {
                                                        i = R.id.iplay_layout_vibrate_arrow_up;
                                                        FrameLayout frameLayout7 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                        if (frameLayout7 != null) {
                                                            i = R.id.iplay_layout_vibrate_btn;
                                                            FrameLayout frameLayout8 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                            if (frameLayout8 != null) {
                                                                i = R.id.iplay_layout_vibration;
                                                                FrameLayout frameLayout9 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                if (frameLayout9 != null) {
                                                                    i = R.id.iplay_layout_vibration_border;
                                                                    LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (linearLayout2 != null) {
                                                                        i = R.id.iplay_playView;
                                                                        PlayView_Four playView_Four = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                                                        if (playView_Four != null) {
                                                                            i = R.id.iplay_progress_speed;
                                                                            VerticalProgressBar verticalProgressBar = (VerticalProgressBar) ViewBindings.findChildViewById(view, i);
                                                                            if (verticalProgressBar != null) {
                                                                                i = R.id.iplay_recycler_mode;
                                                                                RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                if (recyclerView != null) {
                                                                                    i = R.id.iplay_recycler_vibration;
                                                                                    RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                    if (recyclerView2 != null) {
                                                                                        i = R.id.iplay_txt_min_speed;
                                                                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView != null) {
                                                                                            i = R.id.iplay_txt_speed;
                                                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView2 != null) {
                                                                                                i = R.id.iplay_txt_speed_safe;
                                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView3 != null) {
                                                                                                    i = R.id.layout_action;
                                                                                                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                                                    if (constraintLayout != null) {
                                                                                                        i = R.id.title_txt_status;
                                                                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView4 != null) {
                                                                                                            return new FragmentRemoteSmBinding((LinearLayout) view, imageView, imageView2, imageView3, imageView4, frameLayout, frameLayout2, imageView5, frameLayout3, linearLayout, frameLayout4, frameLayout5, frameLayout6, frameLayout7, frameLayout8, frameLayout9, linearLayout2, playView_Four, verticalProgressBar, recyclerView, recyclerView2, textView, textView2, textView3, constraintLayout, textView4);
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

    public static FragmentRemoteSmBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentRemoteSmBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_remote_sm, viewGroup, false);
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
