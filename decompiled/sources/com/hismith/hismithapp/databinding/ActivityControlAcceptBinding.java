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
public final class ActivityControlAcceptBinding implements ViewBinding {
    public final ConstraintLayout constraintLayout;
    public final ImageView imgAccept;
    public final ImageView imgHead;
    public final CircleImageView imgLogo;
    public final ImageView imgRefuse;
    public final ImageView imgSetting;
    public final ImageView imgSmall;
    public final ImageView iplayLayoutBg;
    public final ConstraintLayout layout;
    public final LinearLayout layoutAccept;
    public final FrameLayout layoutBg;
    public final ConstraintLayout layoutContent;
    public final LinearLayout layoutControl;
    public final FrameLayout layoutFrame;
    public final LinearLayout layoutRefuse;
    public final FrameLayout layoutRoot;
    private final FrameLayout rootView;
    public final ImageView switchHudong;
    public final TextView titleTxtName;
    public final TextView titleTxtStatus;
    public final TextView txtAccept;
    public final TextView txtRefuse;
    public final TextView txtRemote;

    private ActivityControlAcceptBinding(FrameLayout frameLayout, ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ConstraintLayout constraintLayout2, LinearLayout linearLayout, FrameLayout frameLayout2, ConstraintLayout constraintLayout3, LinearLayout linearLayout2, FrameLayout frameLayout3, LinearLayout linearLayout3, FrameLayout frameLayout4, ImageView imageView7, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = frameLayout;
        this.constraintLayout = constraintLayout;
        this.imgAccept = imageView;
        this.imgHead = imageView2;
        this.imgLogo = circleImageView;
        this.imgRefuse = imageView3;
        this.imgSetting = imageView4;
        this.imgSmall = imageView5;
        this.iplayLayoutBg = imageView6;
        this.layout = constraintLayout2;
        this.layoutAccept = linearLayout;
        this.layoutBg = frameLayout2;
        this.layoutContent = constraintLayout3;
        this.layoutControl = linearLayout2;
        this.layoutFrame = frameLayout3;
        this.layoutRefuse = linearLayout3;
        this.layoutRoot = frameLayout4;
        this.switchHudong = imageView7;
        this.titleTxtName = textView;
        this.titleTxtStatus = textView2;
        this.txtAccept = textView3;
        this.txtRefuse = textView4;
        this.txtRemote = textView5;
    }

    public static ActivityControlAcceptBinding bind(View view) {
        int i = R.id.constraintLayout;
        ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
        if (constraintLayout != null) {
            i = R.id.img_accept;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_head;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.img_logo;
                    CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                    if (circleImageView != null) {
                        i = R.id.img_refuse;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.img_setting;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.img_small;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.iplay_layout_bg;
                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView6 != null) {
                                        i = R.id.layout;
                                        ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                        if (constraintLayout2 != null) {
                                            i = R.id.layout_accept;
                                            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout != null) {
                                                i = R.id.layout_bg;
                                                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                if (frameLayout != null) {
                                                    i = R.id.layout_content;
                                                    ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                    if (constraintLayout3 != null) {
                                                        i = R.id.layout_control;
                                                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                        if (linearLayout2 != null) {
                                                            i = R.id.layout_frame;
                                                            FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                            if (frameLayout2 != null) {
                                                                i = R.id.layout_refuse;
                                                                LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                if (linearLayout3 != null) {
                                                                    FrameLayout frameLayout3 = (FrameLayout) view;
                                                                    i = R.id.switch_hudong;
                                                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                    if (imageView7 != null) {
                                                                        i = R.id.title_txt_name;
                                                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView != null) {
                                                                            i = R.id.title_txt_status;
                                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView2 != null) {
                                                                                i = R.id.txt_accept;
                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView3 != null) {
                                                                                    i = R.id.txt_refuse;
                                                                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView4 != null) {
                                                                                        i = R.id.txt_remote;
                                                                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView5 != null) {
                                                                                            return new ActivityControlAcceptBinding(frameLayout3, constraintLayout, imageView, imageView2, circleImageView, imageView3, imageView4, imageView5, imageView6, constraintLayout2, linearLayout, frameLayout, constraintLayout3, linearLayout2, frameLayout2, linearLayout3, frameLayout3, imageView7, textView, textView2, textView3, textView4, textView5);
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

    public static ActivityControlAcceptBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityControlAcceptBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_control_accept, viewGroup, false);
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
