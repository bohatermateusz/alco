package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Group;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.GradientRoundProgress;

/* loaded from: classes3.dex */
public final class PlugPiuTimerBinding implements ViewBinding {
    public final Group groupPiu;
    public final ImageView imgBattery;
    public final ImageView imgPiuBg;
    public final ImageView imgPiuTime;
    public final ImageView iplayImgPiu;
    public final FrameLayout iplayLayoutPiuBtn;
    public final GradientRoundProgress iplayProgressPiu;
    public final ConstraintLayout layoutBattery;
    private final ConstraintLayout rootView;
    public final AppCompatTextView txtPiuTime;
    public final View viewBattery;

    private PlugPiuTimerBinding(ConstraintLayout constraintLayout, Group group, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, FrameLayout frameLayout, GradientRoundProgress gradientRoundProgress, ConstraintLayout constraintLayout2, AppCompatTextView appCompatTextView, View view) {
        this.rootView = constraintLayout;
        this.groupPiu = group;
        this.imgBattery = imageView;
        this.imgPiuBg = imageView2;
        this.imgPiuTime = imageView3;
        this.iplayImgPiu = imageView4;
        this.iplayLayoutPiuBtn = frameLayout;
        this.iplayProgressPiu = gradientRoundProgress;
        this.layoutBattery = constraintLayout2;
        this.txtPiuTime = appCompatTextView;
        this.viewBattery = view;
    }

    public static PlugPiuTimerBinding bind(View view) {
        View findChildViewById;
        int i = R.id.group_piu;
        Group group = (Group) ViewBindings.findChildViewById(view, i);
        if (group != null) {
            i = R.id.img_battery;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_piu_bg;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.img_piu_time;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.iplay_img_piu;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.iplay_layout_piu_btn;
                            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                            if (frameLayout != null) {
                                i = R.id.iplay_progress_piu;
                                GradientRoundProgress gradientRoundProgress = (GradientRoundProgress) ViewBindings.findChildViewById(view, i);
                                if (gradientRoundProgress != null) {
                                    i = R.id.layout_battery;
                                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                    if (constraintLayout != null) {
                                        i = R.id.txt_piu_time;
                                        AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                        if (appCompatTextView != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_battery))) != null) {
                                            return new PlugPiuTimerBinding((ConstraintLayout) view, group, imageView, imageView2, imageView3, imageView4, frameLayout, gradientRoundProgress, constraintLayout, appCompatTextView, findChildViewById);
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

    public static PlugPiuTimerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static PlugPiuTimerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.plug_piu_timer, viewGroup, false);
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
