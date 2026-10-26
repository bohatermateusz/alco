package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public final class FragmentCcSmvBinding implements ViewBinding {
    public final BothWayArrowView arrowBoth;
    public final ImageView imgGesture;
    public final ImageView iplayImgCloseMode;
    public final ImageView iplayImgCloseVibration;
    public final ImageView iplayImgHeart;
    public final ImageView iplayImgMic;
    public final ImageView iplayImgMicVibrate;
    public final ImageView iplayImgMode;
    public final ImageView iplayImgShake;
    public final ImageView iplayImgShakeVibrate;
    public final ImageView iplayImgVibrate;
    public final FrameLayout iplayLayoutArrowDown;
    public final FrameLayout iplayLayoutArrowUp;
    public final ImageView iplayLayoutBg;
    public final FrameLayout iplayLayoutMicBtn;
    public final FrameLayout iplayLayoutMicVibrateBtn;
    public final FrameLayout iplayLayoutMode;
    public final LinearLayout iplayLayoutModeBorder;
    public final FrameLayout iplayLayoutModeBtn;
    public final FrameLayout iplayLayoutShakeBtn;
    public final FrameLayout iplayLayoutShakeVibrateBtn;
    public final FrameLayout iplayLayoutVibrateArrowDown;
    public final FrameLayout iplayLayoutVibrateArrowUp;
    public final FrameLayout iplayLayoutVibrateBtn;
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
    public final TextView iplayTxtVibrate;
    public final LinearLayout layoutArrow;
    public final RelativeLayout layoutDeviceInfo;
    private final LinearLayout rootView;
    public final TextView txtSliding;
    public final View viewTouch;

    private FragmentCcSmvBinding(LinearLayout linearLayout, BothWayArrowView bothWayArrowView, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView11, FrameLayout frameLayout3, FrameLayout frameLayout4, FrameLayout frameLayout5, LinearLayout linearLayout2, FrameLayout frameLayout6, FrameLayout frameLayout7, FrameLayout frameLayout8, FrameLayout frameLayout9, FrameLayout frameLayout10, FrameLayout frameLayout11, FrameLayout frameLayout12, LinearLayout linearLayout3, PlayView_Four playView_Four, PlayView_Four playView_Four2, VerticalProgressBar verticalProgressBar, VerticalProgressBar verticalProgressBar2, RecyclerView recyclerView, RecyclerView recyclerView2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, LinearLayout linearLayout4, RelativeLayout relativeLayout, TextView textView7, View view) {
        this.rootView = linearLayout;
        this.arrowBoth = bothWayArrowView;
        this.imgGesture = imageView;
        this.iplayImgCloseMode = imageView2;
        this.iplayImgCloseVibration = imageView3;
        this.iplayImgHeart = imageView4;
        this.iplayImgMic = imageView5;
        this.iplayImgMicVibrate = imageView6;
        this.iplayImgMode = imageView7;
        this.iplayImgShake = imageView8;
        this.iplayImgShakeVibrate = imageView9;
        this.iplayImgVibrate = imageView10;
        this.iplayLayoutArrowDown = frameLayout;
        this.iplayLayoutArrowUp = frameLayout2;
        this.iplayLayoutBg = imageView11;
        this.iplayLayoutMicBtn = frameLayout3;
        this.iplayLayoutMicVibrateBtn = frameLayout4;
        this.iplayLayoutMode = frameLayout5;
        this.iplayLayoutModeBorder = linearLayout2;
        this.iplayLayoutModeBtn = frameLayout6;
        this.iplayLayoutShakeBtn = frameLayout7;
        this.iplayLayoutShakeVibrateBtn = frameLayout8;
        this.iplayLayoutVibrateArrowDown = frameLayout9;
        this.iplayLayoutVibrateArrowUp = frameLayout10;
        this.iplayLayoutVibrateBtn = frameLayout11;
        this.iplayLayoutVibration = frameLayout12;
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
        this.iplayTxtVibrate = textView6;
        this.layoutArrow = linearLayout4;
        this.layoutDeviceInfo = relativeLayout;
        this.txtSliding = textView7;
        this.viewTouch = view;
    }

    public static FragmentCcSmvBinding bind(View view) {
        View findChildViewById;
        int i = R.id.arrow_both;
        BothWayArrowView bothWayArrowView = (BothWayArrowView) ViewBindings.findChildViewById(view, i);
        if (bothWayArrowView != null) {
            i = R.id.img_gesture;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.iplay_img_close_mode;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.iplay_img_close_vibration;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.iplay_img_heart;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.iplay_img_mic;
                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView5 != null) {
                                i = R.id.iplay_img_mic_vibrate;
                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView6 != null) {
                                    i = R.id.iplay_img_mode;
                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView7 != null) {
                                        i = R.id.iplay_img_shake;
                                        ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView8 != null) {
                                            i = R.id.iplay_img_shake_vibrate;
                                            ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView9 != null) {
                                                i = R.id.iplay_img_vibrate;
                                                ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView10 != null) {
                                                    i = R.id.iplay_layout_arrow_down;
                                                    FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                    if (frameLayout != null) {
                                                        i = R.id.iplay_layout_arrow_up;
                                                        FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                        if (frameLayout2 != null) {
                                                            i = R.id.iplay_layout_bg;
                                                            ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView11 != null) {
                                                                i = R.id.iplay_layout_mic_btn;
                                                                FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                if (frameLayout3 != null) {
                                                                    i = R.id.iplay_layout_mic_vibrate_btn;
                                                                    FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (frameLayout4 != null) {
                                                                        i = R.id.iplay_layout_mode;
                                                                        FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (frameLayout5 != null) {
                                                                            i = R.id.iplay_layout_mode_border;
                                                                            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                            if (linearLayout != null) {
                                                                                i = R.id.iplay_layout_mode_btn;
                                                                                FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                if (frameLayout6 != null) {
                                                                                    i = R.id.iplay_layout_shake_btn;
                                                                                    FrameLayout frameLayout7 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                    if (frameLayout7 != null) {
                                                                                        i = R.id.iplay_layout_shake_vibrate_btn;
                                                                                        FrameLayout frameLayout8 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                        if (frameLayout8 != null) {
                                                                                            i = R.id.iplay_layout_vibrate_arrow_down;
                                                                                            FrameLayout frameLayout9 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                            if (frameLayout9 != null) {
                                                                                                i = R.id.iplay_layout_vibrate_arrow_up;
                                                                                                FrameLayout frameLayout10 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                if (frameLayout10 != null) {
                                                                                                    i = R.id.iplay_layout_vibrate_btn;
                                                                                                    FrameLayout frameLayout11 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                    if (frameLayout11 != null) {
                                                                                                        i = R.id.iplay_layout_vibration;
                                                                                                        FrameLayout frameLayout12 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                        if (frameLayout12 != null) {
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
                                                                                                                                                            i = R.id.iplay_txt_vibrate;
                                                                                                                                                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                            if (textView6 != null) {
                                                                                                                                                                i = R.id.layout_arrow;
                                                                                                                                                                LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                if (linearLayout3 != null) {
                                                                                                                                                                    i = R.id.layout_device_info;
                                                                                                                                                                    RelativeLayout relativeLayout = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                    if (relativeLayout != null) {
                                                                                                                                                                        i = R.id.txt_sliding;
                                                                                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                        if (textView7 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_touch))) != null) {
                                                                                                                                                                            return new FragmentCcSmvBinding((LinearLayout) view, bothWayArrowView, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, imageView8, imageView9, imageView10, frameLayout, frameLayout2, imageView11, frameLayout3, frameLayout4, frameLayout5, linearLayout, frameLayout6, frameLayout7, frameLayout8, frameLayout9, frameLayout10, frameLayout11, frameLayout12, linearLayout2, playView_Four, playView_Four2, verticalProgressBar, verticalProgressBar2, recyclerView, recyclerView2, textView, textView2, textView3, textView4, textView5, textView6, linearLayout3, relativeLayout, textView7, findChildViewById);
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

    public static FragmentCcSmvBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentCcSmvBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_cc_smv, viewGroup, false);
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
