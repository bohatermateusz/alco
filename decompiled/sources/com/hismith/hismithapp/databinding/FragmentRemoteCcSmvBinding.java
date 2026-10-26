package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public final class FragmentRemoteCcSmvBinding implements ViewBinding {
    public final ImageView iplayImgMode;
    public final ImageView iplayImgVibrate;
    public final FrameLayout iplayLayoutArrowDown;
    public final FrameLayout iplayLayoutArrowUp;
    public final ImageView iplayLayoutBg;
    public final FrameLayout iplayLayoutMode;
    public final LinearLayout iplayLayoutModeBorder;
    public final FrameLayout iplayLayoutSpeedSafe;
    public final FrameLayout iplayLayoutVibrateArrowDown;
    public final FrameLayout iplayLayoutVibrateArrowUp;
    public final FrameLayout iplayLayoutVibrateSafe;
    public final FrameLayout iplayLayoutVibration;
    public final LinearLayout iplayLayoutVibrationBorder;
    public final PlayView_Four iplayPlayView;
    public final PlayView_Four iplayPlayViewVibrate;
    public final VerticalProgressBar iplayProgressSpeed;
    public final VerticalProgressBar iplayProgressVibrate;
    public final RecyclerView iplayRecyclerMode;
    public final RecyclerView iplayRecyclerVibration;
    public final TextView iplayTxtMaxSpeed;
    public final TextView iplayTxtMaxVibrate;
    public final TextView iplayTxtMinSpeed;
    public final TextView iplayTxtMinVibrate;
    public final TextView iplayTxtSpeed;
    public final TextView iplayTxtSpeedSafe;
    public final TextView iplayTxtVibrate;
    public final TextView iplayTxtVibrateSafe;
    private final LinearLayout rootView;

    private FragmentRemoteCcSmvBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView3, FrameLayout frameLayout3, LinearLayout linearLayout2, FrameLayout frameLayout4, FrameLayout frameLayout5, FrameLayout frameLayout6, FrameLayout frameLayout7, FrameLayout frameLayout8, LinearLayout linearLayout3, PlayView_Four playView_Four, PlayView_Four playView_Four2, VerticalProgressBar verticalProgressBar, VerticalProgressBar verticalProgressBar2, RecyclerView recyclerView, RecyclerView recyclerView2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8) {
        this.rootView = linearLayout;
        this.iplayImgMode = imageView;
        this.iplayImgVibrate = imageView2;
        this.iplayLayoutArrowDown = frameLayout;
        this.iplayLayoutArrowUp = frameLayout2;
        this.iplayLayoutBg = imageView3;
        this.iplayLayoutMode = frameLayout3;
        this.iplayLayoutModeBorder = linearLayout2;
        this.iplayLayoutSpeedSafe = frameLayout4;
        this.iplayLayoutVibrateArrowDown = frameLayout5;
        this.iplayLayoutVibrateArrowUp = frameLayout6;
        this.iplayLayoutVibrateSafe = frameLayout7;
        this.iplayLayoutVibration = frameLayout8;
        this.iplayLayoutVibrationBorder = linearLayout3;
        this.iplayPlayView = playView_Four;
        this.iplayPlayViewVibrate = playView_Four2;
        this.iplayProgressSpeed = verticalProgressBar;
        this.iplayProgressVibrate = verticalProgressBar2;
        this.iplayRecyclerMode = recyclerView;
        this.iplayRecyclerVibration = recyclerView2;
        this.iplayTxtMaxSpeed = textView;
        this.iplayTxtMaxVibrate = textView2;
        this.iplayTxtMinSpeed = textView3;
        this.iplayTxtMinVibrate = textView4;
        this.iplayTxtSpeed = textView5;
        this.iplayTxtSpeedSafe = textView6;
        this.iplayTxtVibrate = textView7;
        this.iplayTxtVibrateSafe = textView8;
    }

    public static FragmentRemoteCcSmvBinding bind(View view) {
        int i = R.id.iplay_img_mode;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.iplay_img_vibrate;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.iplay_layout_arrow_down;
                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout != null) {
                    i = R.id.iplay_layout_arrow_up;
                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                    if (frameLayout2 != null) {
                        i = R.id.iplay_layout_bg;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.iplay_layout_mode;
                            FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                            if (frameLayout3 != null) {
                                i = R.id.iplay_layout_mode_border;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.iplay_layout_speed_safe;
                                    FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                    if (frameLayout4 != null) {
                                        i = R.id.iplay_layout_vibrate_arrow_down;
                                        FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                        if (frameLayout5 != null) {
                                            i = R.id.iplay_layout_vibrate_arrow_up;
                                            FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                            if (frameLayout6 != null) {
                                                i = R.id.iplay_layout_vibrate_safe;
                                                FrameLayout frameLayout7 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                if (frameLayout7 != null) {
                                                    i = R.id.iplay_layout_vibration;
                                                    FrameLayout frameLayout8 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                    if (frameLayout8 != null) {
                                                        i = R.id.iplay_layout_vibration_border;
                                                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                        if (linearLayout2 != null) {
                                                            i = R.id.iplay_playView;
                                                            PlayView_Four playView_Four = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                                            if (playView_Four != null) {
                                                                i = R.id.iplay_playView_vibrate;
                                                                PlayView_Four playView_Four2 = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                                                if (playView_Four2 != null) {
                                                                    i = R.id.iplay_progress_speed;
                                                                    VerticalProgressBar verticalProgressBar = (VerticalProgressBar) ViewBindings.findChildViewById(view, i);
                                                                    if (verticalProgressBar != null) {
                                                                        i = R.id.iplay_progress_vibrate;
                                                                        VerticalProgressBar verticalProgressBar2 = (VerticalProgressBar) ViewBindings.findChildViewById(view, i);
                                                                        if (verticalProgressBar2 != null) {
                                                                            i = R.id.iplay_recycler_mode;
                                                                            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                            if (recyclerView != null) {
                                                                                i = R.id.iplay_recycler_vibration;
                                                                                RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                if (recyclerView2 != null) {
                                                                                    i = R.id.iplay_txt_max_speed;
                                                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView != null) {
                                                                                        i = R.id.iplay_txt_max_vibrate;
                                                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView2 != null) {
                                                                                            i = R.id.iplay_txt_min_speed;
                                                                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView3 != null) {
                                                                                                i = R.id.iplay_txt_min_vibrate;
                                                                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView4 != null) {
                                                                                                    i = R.id.iplay_txt_speed;
                                                                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView5 != null) {
                                                                                                        i = R.id.iplay_txt_speed_safe;
                                                                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView6 != null) {
                                                                                                            i = R.id.iplay_txt_vibrate;
                                                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (textView7 != null) {
                                                                                                                i = R.id.iplay_txt_vibrate_safe;
                                                                                                                TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView8 != null) {
                                                                                                                    return new FragmentRemoteCcSmvBinding((LinearLayout) view, imageView, imageView2, frameLayout, frameLayout2, imageView3, frameLayout3, linearLayout, frameLayout4, frameLayout5, frameLayout6, frameLayout7, frameLayout8, linearLayout2, playView_Four, playView_Four2, verticalProgressBar, verticalProgressBar2, recyclerView, recyclerView2, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8);
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

    public static FragmentRemoteCcSmvBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentRemoteCcSmvBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_remote_cc_smv, viewGroup, false);
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
