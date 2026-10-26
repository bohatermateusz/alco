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
public final class FragmentIplayNewUiBinding implements ViewBinding {
    public final ImageView imgCloseLeft;
    public final ImageView imgCloseRight;
    public final ImageView imgRunMode;
    public final ImageView iplayImgCloseMode;
    public final ImageView iplayImgCloseVibration;
    public final ImageView iplayImgMode;
    public final ImageView iplayImgPosition;
    public final RelativeLayout iplayLayout;
    public final FrameLayout iplayLayoutArrowDown;
    public final FrameLayout iplayLayoutArrowUp;
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

    private FragmentIplayNewUiBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, RelativeLayout relativeLayout, FrameLayout frameLayout, FrameLayout frameLayout2, FrameLayout frameLayout3, LinearLayout linearLayout2, FrameLayout frameLayout4, FrameLayout frameLayout5, FrameLayout frameLayout6, FrameLayout frameLayout7, FrameLayout frameLayout8, LinearLayout linearLayout3, PlayView_Four playView_Four, RecyclerView recyclerView, RecyclerView recyclerView2, FrameLayout frameLayout9, FrameLayout frameLayout10, FrameLayout frameLayout11, ProgressBar progressBar, ProgressBar progressBar2, CustomSeekbar customSeekbar, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, AppCompatTextView appCompatTextView, View view) {
        this.rootView = linearLayout;
        this.imgCloseLeft = imageView;
        this.imgCloseRight = imageView2;
        this.imgRunMode = imageView3;
        this.iplayImgCloseMode = imageView4;
        this.iplayImgCloseVibration = imageView5;
        this.iplayImgMode = imageView6;
        this.iplayImgPosition = imageView7;
        this.iplayLayout = relativeLayout;
        this.iplayLayoutArrowDown = frameLayout;
        this.iplayLayoutArrowUp = frameLayout2;
        this.iplayLayoutMode = frameLayout3;
        this.iplayLayoutModeBorder = linearLayout2;
        this.iplayLayoutModeBtn = frameLayout4;
        this.iplayLayoutTouchPosition = frameLayout5;
        this.iplayLayoutVibrateArrowDown = frameLayout6;
        this.iplayLayoutVibrateArrowUp = frameLayout7;
        this.iplayLayoutVibration = frameLayout8;
        this.iplayLayoutVibrationBorder = linearLayout3;
        this.iplayPlayView = playView_Four;
        this.iplayRecyclerMode = recyclerView;
        this.iplayRecyclerVibration = recyclerView2;
        this.layoutPositionMax = frameLayout9;
        this.layoutPositionMin = frameLayout10;
        this.layoutSpeedMax = frameLayout11;
        this.progressPosition = progressBar;
        this.progressSpeed = progressBar2;
        this.seekSmooth = customSeekbar;
        this.txtPosition = textView;
        this.txtPositionLabel = textView2;
        this.txtPositionMax = textView3;
        this.txtPositionMin = textView4;
        this.txtSmooth = textView5;
        this.txtSmoothLabel = textView6;
        this.txtSpeed = textView7;
        this.txtSpeedLabel = textView8;
        this.txtSpeedMax = appCompatTextView;
        this.viewTouch = view;
    }

    public static FragmentIplayNewUiBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_close_left;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_close_right;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_run_mode;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.iplay_img_close_mode;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.iplay_img_close_vibration;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.iplay_img_mode;
                            ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView6 != null) {
                                i = R.id.iplay_img_position;
                                ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView7 != null) {
                                    i = R.id.iplay_layout;
                                    RelativeLayout relativeLayout = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                                    if (relativeLayout != null) {
                                        i = R.id.iplay_layout_arrow_down;
                                        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                        if (frameLayout != null) {
                                            i = R.id.iplay_layout_arrow_up;
                                            FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                            if (frameLayout2 != null) {
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
                                                                i = R.id.iplay_layout_vibrate_arrow_down;
                                                                FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                if (frameLayout6 != null) {
                                                                    i = R.id.iplay_layout_vibrate_arrow_up;
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
                                                                                    i = R.id.iplay_recycler_mode;
                                                                                    RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                    if (recyclerView != null) {
                                                                                        i = R.id.iplay_recycler_vibration;
                                                                                        RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                        if (recyclerView2 != null) {
                                                                                            i = R.id.layout_position_max;
                                                                                            FrameLayout frameLayout9 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                            if (frameLayout9 != null) {
                                                                                                i = R.id.layout_position_min;
                                                                                                FrameLayout frameLayout10 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                if (frameLayout10 != null) {
                                                                                                    i = R.id.layout_speed_max;
                                                                                                    FrameLayout frameLayout11 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                    if (frameLayout11 != null) {
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
                                                                                                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (textView3 != null) {
                                                                                                                                i = R.id.txt_position_min;
                                                                                                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (textView4 != null) {
                                                                                                                                    i = R.id.txt_smooth;
                                                                                                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                    if (textView5 != null) {
                                                                                                                                        i = R.id.txt_smooth_label;
                                                                                                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                        if (textView6 != null) {
                                                                                                                                            i = R.id.txt_speed;
                                                                                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                            if (textView7 != null) {
                                                                                                                                                i = R.id.txt_speed_label;
                                                                                                                                                TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                if (textView8 != null) {
                                                                                                                                                    i = R.id.txt_speed_max;
                                                                                                                                                    AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                    if (appCompatTextView != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_touch))) != null) {
                                                                                                                                                        return new FragmentIplayNewUiBinding((LinearLayout) view, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, relativeLayout, frameLayout, frameLayout2, frameLayout3, linearLayout, frameLayout4, frameLayout5, frameLayout6, frameLayout7, frameLayout8, linearLayout2, playView_Four, recyclerView, recyclerView2, frameLayout9, frameLayout10, frameLayout11, progressBar, progressBar2, customSeekbar, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, appCompatTextView, findChildViewById);
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

    public static FragmentIplayNewUiBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentIplayNewUiBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_iplay_new_ui, viewGroup, false);
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
