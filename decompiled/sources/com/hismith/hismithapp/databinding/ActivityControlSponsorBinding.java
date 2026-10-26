package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ActivityControlSponsorBinding implements ViewBinding {
    public final ConstraintLayout constraintLayout;
    public final ImageView imgHead;
    public final CircleImageView imgLogo;
    public final ImageView imgRefuse;
    public final ImageView imgSetting;
    public final ImageView imgSmall;
    public final ImageView iplayLayoutBg;
    public final ConstraintLayout layout;
    public final FrameLayout layoutBg;
    public final ConstraintLayout layoutContent;
    public final LinearLayout layoutControl;
    public final FrameLayout layoutFrame;
    public final LinearLayout layoutRefuse;
    public final FrameLayout layoutRoot;
    private final FrameLayout rootView;
    public final TextView titleTxtName;
    public final TextView titleTxtStatus;
    public final TextView txtLog;
    public final TextView txtRefuse;
    public final TextView txtRemote;

    private ActivityControlSponsorBinding(FrameLayout frameLayout, ConstraintLayout constraintLayout, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ConstraintLayout constraintLayout2, FrameLayout frameLayout2, ConstraintLayout constraintLayout3, LinearLayout linearLayout, FrameLayout frameLayout3, LinearLayout linearLayout2, FrameLayout frameLayout4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = frameLayout;
        this.constraintLayout = constraintLayout;
        this.imgHead = imageView;
        this.imgLogo = circleImageView;
        this.imgRefuse = imageView2;
        this.imgSetting = imageView3;
        this.imgSmall = imageView4;
        this.iplayLayoutBg = imageView5;
        this.layout = constraintLayout2;
        this.layoutBg = frameLayout2;
        this.layoutContent = constraintLayout3;
        this.layoutControl = linearLayout;
        this.layoutFrame = frameLayout3;
        this.layoutRefuse = linearLayout2;
        this.layoutRoot = frameLayout4;
        this.titleTxtName = textView;
        this.titleTxtStatus = textView2;
        this.txtLog = textView3;
        this.txtRefuse = textView4;
        this.txtRemote = textView5;
    }

    public static ActivityControlSponsorBinding bind(View view) {
        int i = R.id.constraintLayout;
        ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
        if (constraintLayout != null) {
            i = R.id.img_head;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_logo;
                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                if (circleImageView != null) {
                    i = R.id.img_refuse;
                    ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView2 != null) {
                        i = R.id.img_setting;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.img_small;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.iplay_layout_bg;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.layout;
                                    ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                    if (constraintLayout2 != null) {
                                        i = R.id.layout_bg;
                                        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                        if (frameLayout != null) {
                                            i = R.id.layout_content;
                                            ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                            if (constraintLayout3 != null) {
                                                i = R.id.layout_control;
                                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                if (linearLayout != null) {
                                                    i = R.id.layout_frame;
                                                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                    if (frameLayout2 != null) {
                                                        i = R.id.layout_refuse;
                                                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                        if (linearLayout2 != null) {
                                                            FrameLayout frameLayout3 = (FrameLayout) view;
                                                            i = R.id.title_txt_name;
                                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView != null) {
                                                                i = R.id.title_txt_status;
                                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView2 != null) {
                                                                    i = R.id.txt_log;
                                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView3 != null) {
                                                                        i = R.id.txt_refuse;
                                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView4 != null) {
                                                                            i = R.id.txt_remote;
                                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView5 != null) {
                                                                                return new ActivityControlSponsorBinding(frameLayout3, constraintLayout, imageView, circleImageView, imageView2, imageView3, imageView4, imageView5, constraintLayout2, frameLayout, constraintLayout3, linearLayout, frameLayout2, linearLayout2, frameLayout3, textView, textView2, textView3, textView4, textView5);
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

    public static ActivityControlSponsorBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityControlSponsorBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_control_sponsor, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public FrameLayout getRoot() {
        return this.rootView;
    }
}
