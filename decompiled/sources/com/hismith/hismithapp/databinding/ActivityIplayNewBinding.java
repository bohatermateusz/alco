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
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.PlayView_Four;

/* loaded from: classes3.dex */
public final class ActivityIplayNewBinding implements ViewBinding {
    public final ImageView imgCloseLeft;
    public final ImageView imgCloseRight;
    public final ImageView imgGestureLeft;
    public final ImageView imgGestureRight;
    public final ImageView imgRunMode;
    public final ImageView iplayImgCloseMode;
    public final ImageView iplayImgCloseVibration;
    public final ImageView iplayImgMic;
    public final ImageView iplayImgMode;
    public final ImageView iplayImgPosition;
    public final RelativeLayout iplayLayout;
    public final FrameLayout iplayLayoutArrowDown;
    public final FrameLayout iplayLayoutArrowUp;
    public final ImageView iplayLayoutBg;
    public final FrameLayout iplayLayoutMicBtn;
    public final FrameLayout iplayLayoutMode;
    public final LinearLayout iplayLayoutModeBorder;
    public final FrameLayout iplayLayoutModeBtn;
    public final FrameLayout iplayLayoutTouchPosition;
    public final FrameLayout iplayLayoutVibrateArrowDown;
    public final FrameLayout iplayLayoutVibrateArrowUp;
    public final FrameLayout iplayLayoutVibration;
    public final LinearLayout iplayLayoutVibrationBorder;
    public final PlayView_Four iplayPlayView;
    public final RecyclerView iplayRecyclerMode;
    public final RecyclerView iplayRecyclerVibration;
    public final FrameLayout layoutPositionMax;
    public final FrameLayout layoutPositionMin;
    public final FrameLayout layoutSpeedMax;
    public final ProgressBar progressPosition;
    public final ProgressBar progressSpeed;
    private final LinearLayout rootView;
    public final CustomSeekbar seekSmooth;
    public final TextView titleTxtStatus;
    public final TextView txtGestureTip;
    public final TextView txtPosition;
    public final TextView txtPositionLabel;
    public final TextView txtPositionMax;
    public final TextView txtPositionMin;
    public final TextView txtSmooth;
    public final TextView txtSmoothLabel;
    public final TextView txtSpeed;
    public final TextView txtSpeedLabel;
    public final AppCompatTextView txtSpeedMax;
    public final View viewTouch;

    private ActivityIplayNewBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, RelativeLayout relativeLayout, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView11, FrameLayout frameLayout3, FrameLayout frameLayout4, LinearLayout linearLayout2, FrameLayout frameLayout5, FrameLayout frameLayout6, FrameLayout frameLayout7, FrameLayout frameLayout8, FrameLayout frameLayout9, LinearLayout linearLayout3, PlayView_Four playView_Four, RecyclerView recyclerView, RecyclerView recyclerView2, FrameLayout frameLayout10, FrameLayout frameLayout11, FrameLayout frameLayout12, ProgressBar progressBar, ProgressBar progressBar2, CustomSeekbar customSeekbar, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10, AppCompatTextView appCompatTextView, View view) {
        this.rootView = linearLayout;
        this.imgCloseLeft = imageView;
        this.imgCloseRight = imageView2;
        this.imgGestureLeft = imageView3;
        this.imgGestureRight = imageView4;
        this.imgRunMode = imageView5;
        this.iplayImgCloseMode = imageView6;
        this.iplayImgCloseVibration = imageView7;
        this.iplayImgMic = imageView8;
        this.iplayImgMode = imageView9;
        this.iplayImgPosition = imageView10;
        this.iplayLayout = relativeLayout;
        this.iplayLayoutArrowDown = frameLayout;
        this.iplayLayoutArrowUp = frameLayout2;
        this.iplayLayoutBg = imageView11;
        this.iplayLayoutMicBtn = frameLayout3;
        this.iplayLayoutMode = frameLayout4;
        this.iplayLayoutModeBorder = linearLayout2;
        this.iplayLayoutModeBtn = frameLayout5;
        this.iplayLayoutTouchPosition = frameLayout6;
        this.iplayLayoutVibrateArrowDown = frameLayout7;
        this.iplayLayoutVibrateArrowUp = frameLayout8;
        this.iplayLayoutVibration = frameLayout9;
        this.iplayLayoutVibrationBorder = linearLayout3;
        this.iplayPlayView = playView_Four;
        this.iplayRecyclerMode = recyclerView;
        this.iplayRecyclerVibration = recyclerView2;
        this.layoutPositionMax = frameLayout10;
        this.layoutPositionMin = frameLayout11;
        this.layoutSpeedMax = frameLayout12;
        this.progressPosition = progressBar;
        this.progressSpeed = progressBar2;
        this.seekSmooth = customSeekbar;
        this.titleTxtStatus = textView;
        this.txtGestureTip = textView2;
        this.txtPosition = textView3;
        this.txtPositionLabel = textView4;
        this.txtPositionMax = textView5;
        this.txtPositionMin = textView6;
        this.txtSmooth = textView7;
        this.txtSmoothLabel = textView8;
        this.txtSpeed = textView9;
        this.txtSpeedLabel = textView10;
        this.txtSpeedMax = appCompatTextView;
        this.viewTouch = view;
    }

    public static ActivityIplayNewBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_close_left;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_close_right;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_gesture_left;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.img_gesture_right;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.img_run_mode;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.iplay_img_close_mode;
                            ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView6 != null) {
                                i = R.id.iplay_img_close_vibration;
                                ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView7 != null) {
                                    i = R.id.iplay_img_mic;
                                    ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView8 != null) {
                                        i = R.id.iplay_img_mode;
                                        ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView9 != null) {
                                            i = R.id.iplay_img_position;
                                            ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView10 != null) {
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
                                                            ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView11 != null) {
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
                                                                                i = R.id.iplay_layout_touch_position;
                                                                                FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                if (frameLayout6 != null) {
                                                                                    i = R.id.iplay_layout_vibrate_arrow_down;
                                                                                    FrameLayout frameLayout7 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                    if (frameLayout7 != null) {
                                                                                        i = R.id.iplay_layout_vibrate_arrow_up;
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
                                                                                                        i = R.id.iplay_recycler_mode;
                                                                                                        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (recyclerView != null) {
                                                                                                            i = R.id.iplay_recycler_vibration;
                                                                                                            RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (recyclerView2 != null) {
                                                                                                                i = R.id.layout_position_max;
                                                                                                                FrameLayout frameLayout10 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                if (frameLayout10 != null) {
                                                                                                                    i = R.id.layout_position_min;
                                                                                                                    FrameLayout frameLayout11 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (frameLayout11 != null) {
                                                                                                                        i = R.id.layout_speed_max;
                                                                                                                        FrameLayout frameLayout12 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (frameLayout12 != null) {
                                                                                                                            i = R.id.progress_position;
                                                                                                                            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (progressBar != null) {
                                                                                                                                i = R.id.progress_speed;
                                                                                                                                ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (progressBar2 != null) {
                                                                                                                                    i = R.id.seek_smooth;
                                                                                                                                    CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
                                                                                                                                    if (customSeekbar != null) {
                                                                                                                                        i = R.id.title_txt_status;
                                                                                                                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                        if (textView != null) {
                                                                                                                                            i = R.id.txt_gesture_tip;
                                                                                                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                            if (textView2 != null) {
                                                                                                                                                i = R.id.txt_position;
                                                                                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                if (textView3 != null) {
                                                                                                                                                    i = R.id.txt_position_label;
                                                                                                                                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                    if (textView4 != null) {
                                                                                                                                                        i = R.id.txt_position_max;
                                                                                                                                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                        if (textView5 != null) {
                                                                                                                                                            i = R.id.txt_position_min;
                                                                                                                                                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                            if (textView6 != null) {
                                                                                                                                                                i = R.id.txt_smooth;
                                                                                                                                                                TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                if (textView7 != null) {
                                                                                                                                                                    i = R.id.txt_smooth_label;
                                                                                                                                                                    TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                    if (textView8 != null) {
                                                                                                                                                                        i = R.id.txt_speed;
                                                                                                                                                                        TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                        if (textView9 != null) {
                                                                                                                                                                            i = R.id.txt_speed_label;
                                                                                                                                                                            TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                            if (textView10 != null) {
                                                                                                                                                                                i = R.id.txt_speed_max;
                                                                                                                                                                                AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                if (appCompatTextView != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_touch))) != null) {
                                                                                                                                                                                    return new ActivityIplayNewBinding((LinearLayout) view, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, imageView8, imageView9, imageView10, relativeLayout, frameLayout, frameLayout2, imageView11, frameLayout3, frameLayout4, linearLayout, frameLayout5, frameLayout6, frameLayout7, frameLayout8, frameLayout9, linearLayout2, playView_Four, recyclerView, recyclerView2, frameLayout10, frameLayout11, frameLayout12, progressBar, progressBar2, customSeekbar, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9, textView10, appCompatTextView, findChildViewById);
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
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityIplayNewBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityIplayNewBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_iplay_new, viewGroup, false);
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
