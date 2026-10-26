package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public final class FragmentHudongSmYouBinding implements ViewBinding {
    public final BothWayArrowView arrowBoth;
    public final ImageView imgGesture;
    public final ImageView imgReversal;
    public final ImageView imgSpeedMode;
    public final ImageView imgSpeedModeBg;
    public final ImageView imgVibrateMode;
    public final ImageView imgVibrateModeBg;
    public final ImageView iplayImgHeart;
    public final LinearLayout layoutArrow;
    public final ConstraintLayout layoutRoot;
    public final ConstraintLayout layoutSpeedMode;
    public final ConstraintLayout layoutVibrateMode;
    public final View line1;
    public final PlayView_Four playView;
    public final VerticalProgressBar progressSpeed;
    private final ConstraintLayout rootView;
    public final TextView txtSliding;
    public final View viewTouch;
    public final CustomWebView webDevice;

    private FragmentHudongSmYouBinding(ConstraintLayout constraintLayout, BothWayArrowView bothWayArrowView, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, LinearLayout linearLayout, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, ConstraintLayout constraintLayout4, View view, PlayView_Four playView_Four, VerticalProgressBar verticalProgressBar, TextView textView, View view2, CustomWebView customWebView) {
        this.rootView = constraintLayout;
        this.arrowBoth = bothWayArrowView;
        this.imgGesture = imageView;
        this.imgReversal = imageView2;
        this.imgSpeedMode = imageView3;
        this.imgSpeedModeBg = imageView4;
        this.imgVibrateMode = imageView5;
        this.imgVibrateModeBg = imageView6;
        this.iplayImgHeart = imageView7;
        this.layoutArrow = linearLayout;
        this.layoutRoot = constraintLayout2;
        this.layoutSpeedMode = constraintLayout3;
        this.layoutVibrateMode = constraintLayout4;
        this.line1 = view;
        this.playView = playView_Four;
        this.progressSpeed = verticalProgressBar;
        this.txtSliding = textView;
        this.viewTouch = view2;
        this.webDevice = customWebView;
    }

    public static FragmentHudongSmYouBinding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        int i = R.id.arrow_both;
        BothWayArrowView bothWayArrowView = (BothWayArrowView) ViewBindings.findChildViewById(view, i);
        if (bothWayArrowView != null) {
            i = R.id.img_gesture;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_reversal;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.img_speed_mode;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.img_speed_mode_bg;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.img_vibrate_mode;
                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView5 != null) {
                                i = R.id.img_vibrate_mode_bg;
                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView6 != null) {
                                    i = R.id.iplay_img_heart;
                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView7 != null) {
                                        i = R.id.layout_arrow;
                                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout != null) {
                                            ConstraintLayout constraintLayout = (ConstraintLayout) view;
                                            i = R.id.layout_speed_mode;
                                            ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                            if (constraintLayout2 != null) {
                                                i = R.id.layout_vibrate_mode;
                                                ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                if (constraintLayout3 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line1))) != null) {
                                                    i = R.id.playView;
                                                    PlayView_Four playView_Four = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                                    if (playView_Four != null) {
                                                        i = R.id.progress_speed;
                                                        VerticalProgressBar verticalProgressBar = (VerticalProgressBar) ViewBindings.findChildViewById(view, i);
                                                        if (verticalProgressBar != null) {
                                                            i = R.id.txt_sliding;
                                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.view_touch))) != null) {
                                                                i = R.id.web_device;
                                                                CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                                                if (customWebView != null) {
                                                                    return new FragmentHudongSmYouBinding(constraintLayout, bothWayArrowView, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, linearLayout, constraintLayout, constraintLayout2, constraintLayout3, findChildViewById, playView_Four, verticalProgressBar, textView, findChildViewById2, customWebView);
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

    public static FragmentHudongSmYouBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentHudongSmYouBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_hudong_sm_you, viewGroup, false);
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
