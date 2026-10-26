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
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public final class ActivityIplayBinding implements ViewBinding {
    public final BothWayArrowView arrowBoth;
    public final ImageView imgGesture;
    public final ImageView iplayImgCloseLeft;
    public final ImageView iplayImgCloseMode;
    public final ImageView iplayImgCloseRight;
    public final ImageView iplayImgCloseVibration;
    public final ImageView iplayImgGestureLeft;
    public final ImageView iplayImgGestureRight;
    public final ImageView iplayImgHeart;
    public final ImageView iplayImgMic;
    public final ImageView iplayImgMode;
    public final ImageView iplayImgMusic;
    public final ImageView iplayImgShake;
    public final ImageView iplayImgVibrate;
    public final FrameLayout iplayLayoutArrowDown;
    public final FrameLayout iplayLayoutArrowUp;
    public final ImageView iplayLayoutBg;
    public final FrameLayout iplayLayoutMicBtn;
    public final FrameLayout iplayLayoutMode;
    public final LinearLayout iplayLayoutModeBorder;
    public final FrameLayout iplayLayoutModeBtn;
    public final FrameLayout iplayLayoutMusicBtn;
    public final FrameLayout iplayLayoutShakeBtn;
    public final FrameLayout iplayLayoutVibrateArrowDown;
    public final FrameLayout iplayLayoutVibrateArrowUp;
    public final FrameLayout iplayLayoutVibrateBtn;
    public final FrameLayout iplayLayoutVibration;
    public final LinearLayout iplayLayoutVibrationBorder;
    public final PlayView_Four iplayPlayView;
    public final VerticalProgressBar iplayProgressSpeed;
    public final RecyclerView iplayRecyclerMode;
    public final RecyclerView iplayRecyclerVibration;
    public final TextView iplayTxtGestureTip;
    public final TextView iplayTxtMaxSpeed;
    public final TextView iplayTxtMinSpeed;
    public final TextView iplayTxtSpeed;
    public final LinearLayout layoutArrow;
    private final LinearLayout rootView;
    public final TextView titleTxtName;
    public final TextView titleTxtStatus;
    public final TextView txtSliding;
    public final View viewTouch;

    private ActivityIplayBinding(LinearLayout linearLayout, BothWayArrowView bothWayArrowView, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, ImageView imageView11, ImageView imageView12, ImageView imageView13, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView14, FrameLayout frameLayout3, FrameLayout frameLayout4, LinearLayout linearLayout2, FrameLayout frameLayout5, FrameLayout frameLayout6, FrameLayout frameLayout7, FrameLayout frameLayout8, FrameLayout frameLayout9, FrameLayout frameLayout10, FrameLayout frameLayout11, LinearLayout linearLayout3, PlayView_Four playView_Four, VerticalProgressBar verticalProgressBar, RecyclerView recyclerView, RecyclerView recyclerView2, TextView textView, TextView textView2, TextView textView3, TextView textView4, LinearLayout linearLayout4, TextView textView5, TextView textView6, TextView textView7, View view) {
        this.rootView = linearLayout;
        this.arrowBoth = bothWayArrowView;
        this.imgGesture = imageView;
        this.iplayImgCloseLeft = imageView2;
        this.iplayImgCloseMode = imageView3;
        this.iplayImgCloseRight = imageView4;
        this.iplayImgCloseVibration = imageView5;
        this.iplayImgGestureLeft = imageView6;
        this.iplayImgGestureRight = imageView7;
        this.iplayImgHeart = imageView8;
        this.iplayImgMic = imageView9;
        this.iplayImgMode = imageView10;
        this.iplayImgMusic = imageView11;
        this.iplayImgShake = imageView12;
        this.iplayImgVibrate = imageView13;
        this.iplayLayoutArrowDown = frameLayout;
        this.iplayLayoutArrowUp = frameLayout2;
        this.iplayLayoutBg = imageView14;
        this.iplayLayoutMicBtn = frameLayout3;
        this.iplayLayoutMode = frameLayout4;
        this.iplayLayoutModeBorder = linearLayout2;
        this.iplayLayoutModeBtn = frameLayout5;
        this.iplayLayoutMusicBtn = frameLayout6;
        this.iplayLayoutShakeBtn = frameLayout7;
        this.iplayLayoutVibrateArrowDown = frameLayout8;
        this.iplayLayoutVibrateArrowUp = frameLayout9;
        this.iplayLayoutVibrateBtn = frameLayout10;
        this.iplayLayoutVibration = frameLayout11;
        this.iplayLayoutVibrationBorder = linearLayout3;
        this.iplayPlayView = playView_Four;
        this.iplayProgressSpeed = verticalProgressBar;
        this.iplayRecyclerMode = recyclerView;
        this.iplayRecyclerVibration = recyclerView2;
        this.iplayTxtGestureTip = textView;
        this.iplayTxtMaxSpeed = textView2;
        this.iplayTxtMinSpeed = textView3;
        this.iplayTxtSpeed = textView4;
        this.layoutArrow = linearLayout4;
        this.titleTxtName = textView5;
        this.titleTxtStatus = textView6;
        this.txtSliding = textView7;
        this.viewTouch = view;
    }

    public static ActivityIplayBinding bind(View view) {
        View findChildViewById;
        int i = R.id.arrow_both;
        BothWayArrowView bothWayArrowView = (BothWayArrowView) ViewBindings.findChildViewById(view, i);
        if (bothWayArrowView != null) {
            i = R.id.img_gesture;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.iplay_img_close_left;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.iplay_img_close_mode;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.iplay_img_close_right;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.iplay_img_close_vibration;
                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView5 != null) {
                                i = R.id.iplay_img_gesture_left;
                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView6 != null) {
                                    i = R.id.iplay_img_gesture_right;
                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView7 != null) {
                                        i = R.id.iplay_img_heart;
                                        ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView8 != null) {
                                            i = R.id.iplay_img_mic;
                                            ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView9 != null) {
                                                i = R.id.iplay_img_mode;
                                                ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView10 != null) {
                                                    i = R.id.iplay_img_music;
                                                    ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                    if (imageView11 != null) {
                                                        i = R.id.iplay_img_shake;
                                                        ImageView imageView12 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                        if (imageView12 != null) {
                                                            i = R.id.iplay_img_vibrate;
                                                            ImageView imageView13 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView13 != null) {
                                                                i = R.id.iplay_layout_arrow_down;
                                                                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                if (frameLayout != null) {
                                                                    i = R.id.iplay_layout_arrow_up;
                                                                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (frameLayout2 != null) {
                                                                        i = R.id.iplay_layout_bg;
                                                                        ImageView imageView14 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                        if (imageView14 != null) {
                                                                            i = R.id.iplay_layout_mic_btn;
                                                                            FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                            if (frameLayout3 != null) {
                                                                                i = R.id.iplay_layout_mode;
                                                                                FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                if (frameLayout4 != null) {
                                                                                    i = R.id.iplay_layout_mode_border;
                                                                                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                    if (linearLayout != null) {
                                                                                        i = R.id.iplay_layout_mode_btn;
                                                                                        FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                        if (frameLayout5 != null) {
                                                                                            i = R.id.iplay_layout_music_btn;
                                                                                            FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                            if (frameLayout6 != null) {
                                                                                                i = R.id.iplay_layout_shake_btn;
                                                                                                FrameLayout frameLayout7 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                if (frameLayout7 != null) {
                                                                                                    i = R.id.iplay_layout_vibrate_arrow_down;
                                                                                                    FrameLayout frameLayout8 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                    if (frameLayout8 != null) {
                                                                                                        i = R.id.iplay_layout_vibrate_arrow_up;
                                                                                                        FrameLayout frameLayout9 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                        if (frameLayout9 != null) {
                                                                                                            i = R.id.iplay_layout_vibrate_btn;
                                                                                                            FrameLayout frameLayout10 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                            if (frameLayout10 != null) {
                                                                                                                i = R.id.iplay_layout_vibration;
                                                                                                                FrameLayout frameLayout11 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                if (frameLayout11 != null) {
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
                                                                                                                                        i = R.id.iplay_txt_gesture_tip;
                                                                                                                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                        if (textView != null) {
                                                                                                                                            i = R.id.iplay_txt_max_speed;
                                                                                                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                            if (textView2 != null) {
                                                                                                                                                i = R.id.iplay_txt_min_speed;
                                                                                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                if (textView3 != null) {
                                                                                                                                                    i = R.id.iplay_txt_speed;
                                                                                                                                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                    if (textView4 != null) {
                                                                                                                                                        i = R.id.layout_arrow;
                                                                                                                                                        LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                                                        if (linearLayout3 != null) {
                                                                                                                                                            i = R.id.title_txt_name;
                                                                                                                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                            if (textView5 != null) {
                                                                                                                                                                i = R.id.title_txt_status;
                                                                                                                                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                if (textView6 != null) {
                                                                                                                                                                    i = R.id.txt_sliding;
                                                                                                                                                                    TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                    if (textView7 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_touch))) != null) {
                                                                                                                                                                        return new ActivityIplayBinding((LinearLayout) view, bothWayArrowView, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, imageView8, imageView9, imageView10, imageView11, imageView12, imageView13, frameLayout, frameLayout2, imageView14, frameLayout3, frameLayout4, linearLayout, frameLayout5, frameLayout6, frameLayout7, frameLayout8, frameLayout9, frameLayout10, frameLayout11, linearLayout2, playView_Four, verticalProgressBar, recyclerView, recyclerView2, textView, textView2, textView3, textView4, linearLayout3, textView5, textView6, textView7, findChildViewById);
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

    public static ActivityIplayBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityIplayBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_iplay, viewGroup, false);
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
