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

/* loaded from: classes3.dex */
public final class DialogPlayVideoPanelBinding implements ViewBinding {
    public final ImageView imgClose;
    public final ImageView imgMotor1;
    public final ImageView imgMotor1Select;
    public final ImageView imgMotor2;
    public final ImageView imgMotor2Select;
    public final FrameLayout layoutDevicePanel;
    public final ConstraintLayout layoutRoot;
    public final LinearLayout layoutTrackContent;
    private final ConstraintLayout rootView;
    public final TextView txtTrackHands;
    public final TextView txtTrackHead;
    public final TextView txtTrackHips;
    public final TextView txtTrackTitle;

    private DialogPlayVideoPanelBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, FrameLayout frameLayout, ConstraintLayout constraintLayout2, LinearLayout linearLayout, TextView textView, TextView textView2, TextView textView3, TextView textView4) {
        this.rootView = constraintLayout;
        this.imgClose = imageView;
        this.imgMotor1 = imageView2;
        this.imgMotor1Select = imageView3;
        this.imgMotor2 = imageView4;
        this.imgMotor2Select = imageView5;
        this.layoutDevicePanel = frameLayout;
        this.layoutRoot = constraintLayout2;
        this.layoutTrackContent = linearLayout;
        this.txtTrackHands = textView;
        this.txtTrackHead = textView2;
        this.txtTrackHips = textView3;
        this.txtTrackTitle = textView4;
    }

    public static DialogPlayVideoPanelBinding bind(View view) {
        int i = R.id.img_close;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_motor1;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_motor1_select;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.img_motor2;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.img_motor2_select;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.layout_device_panel;
                            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                            if (frameLayout != null) {
                                ConstraintLayout constraintLayout = (ConstraintLayout) view;
                                i = R.id.layout_track_content;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.txt_track_hands;
                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView != null) {
                                        i = R.id.txt_track_head;
                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView2 != null) {
                                            i = R.id.txt_track_hips;
                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView3 != null) {
                                                i = R.id.txt_track_title;
                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView4 != null) {
                                                    return new DialogPlayVideoPanelBinding(constraintLayout, imageView, imageView2, imageView3, imageView4, imageView5, frameLayout, constraintLayout, linearLayout, textView, textView2, textView3, textView4);
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

    public static DialogPlayVideoPanelBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogPlayVideoPanelBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_play_video_panel, viewGroup, false);
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
