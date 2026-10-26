package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class FragmentHudongSmsBinding implements ViewBinding {
    public final ImageView imgCloseLeft;
    public final ImageView imgCloseMode;
    public final ImageView imgCloseRight;
    public final ImageView imgCloseVibrateMode;
    public final ImageView imgSpeedMic;
    public final ImageView imgSpeedMode;
    public final ImageView imgSpeedShake;
    public final ImageView imgVibrateMic;
    public final ImageView imgVibrateMode;
    public final ImageView imgVibrateShake;
    public final FrameLayout layoutArrowDown;
    public final FrameLayout layoutArrowUp;
    public final FrameLayout layoutFrameMy;
    public final FrameLayout layoutFrameYou;
    public final ConstraintLayout layoutModeList;
    public final ConstraintLayout layoutRoot;
    public final FrameLayout layoutVibrateArrowDown;
    public final FrameLayout layoutVibrateArrowUp;
    public final ConstraintLayout layoutVibrateModeList;
    public final View line1;
    public final RecyclerView recyclerMode;
    public final RecyclerView recyclerVibrateMode;
    private final ConstraintLayout rootView;

    private FragmentHudongSmsBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, FrameLayout frameLayout, FrameLayout frameLayout2, FrameLayout frameLayout3, FrameLayout frameLayout4, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, FrameLayout frameLayout5, FrameLayout frameLayout6, ConstraintLayout constraintLayout4, View view, RecyclerView recyclerView, RecyclerView recyclerView2) {
        this.rootView = constraintLayout;
        this.imgCloseLeft = imageView;
        this.imgCloseMode = imageView2;
        this.imgCloseRight = imageView3;
        this.imgCloseVibrateMode = imageView4;
        this.imgSpeedMic = imageView5;
        this.imgSpeedMode = imageView6;
        this.imgSpeedShake = imageView7;
        this.imgVibrateMic = imageView8;
        this.imgVibrateMode = imageView9;
        this.imgVibrateShake = imageView10;
        this.layoutArrowDown = frameLayout;
        this.layoutArrowUp = frameLayout2;
        this.layoutFrameMy = frameLayout3;
        this.layoutFrameYou = frameLayout4;
        this.layoutModeList = constraintLayout2;
        this.layoutRoot = constraintLayout3;
        this.layoutVibrateArrowDown = frameLayout5;
        this.layoutVibrateArrowUp = frameLayout6;
        this.layoutVibrateModeList = constraintLayout4;
        this.line1 = view;
        this.recyclerMode = recyclerView;
        this.recyclerVibrateMode = recyclerView2;
    }

    public static FragmentHudongSmsBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_close_left;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_close_mode;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_close_right;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.img_close_vibrate_mode;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.img_speed_mic;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.img_speed_mode;
                            ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView6 != null) {
                                i = R.id.img_speed_shake;
                                ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView7 != null) {
                                    i = R.id.img_vibrate_mic;
                                    ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView8 != null) {
                                        i = R.id.img_vibrate_mode;
                                        ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView9 != null) {
                                            i = R.id.img_vibrate_shake;
                                            ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView10 != null) {
                                                i = R.id.layout_arrow_down;
                                                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                if (frameLayout != null) {
                                                    i = R.id.layout_arrow_up;
                                                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                    if (frameLayout2 != null) {
                                                        i = R.id.layout_frame_my;
                                                        FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                        if (frameLayout3 != null) {
                                                            i = R.id.layout_frame_you;
                                                            FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                            if (frameLayout4 != null) {
                                                                i = R.id.layout_mode_list;
                                                                ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                if (constraintLayout != null) {
                                                                    ConstraintLayout constraintLayout2 = (ConstraintLayout) view;
                                                                    i = R.id.layout_vibrate_arrow_down;
                                                                    FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (frameLayout5 != null) {
                                                                        i = R.id.layout_vibrate_arrow_up;
                                                                        FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (frameLayout6 != null) {
                                                                            i = R.id.layout_vibrate_mode_list;
                                                                            ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                            if (constraintLayout3 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line_1))) != null) {
                                                                                i = R.id.recycler_mode;
                                                                                RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                if (recyclerView != null) {
                                                                                    i = R.id.recycler_vibrate_mode;
                                                                                    RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                    if (recyclerView2 != null) {
                                                                                        return new FragmentHudongSmsBinding(constraintLayout2, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, imageView8, imageView9, imageView10, frameLayout, frameLayout2, frameLayout3, frameLayout4, constraintLayout, constraintLayout2, frameLayout5, frameLayout6, constraintLayout3, findChildViewById, recyclerView, recyclerView2);
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

    public static FragmentHudongSmsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentHudongSmsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sms, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public ConstraintLayout getRoot() {
        return this.rootView;
    }
}
