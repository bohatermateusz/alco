package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.RightDraggableFrameLayout;

/* loaded from: classes3.dex */
public final class ActivityPlaySoloBinding implements ViewBinding {
    public final RightDraggableFrameLayout draglayoutDeviceList;
    public final FrameLayout frameLayout;
    public final FrameLayout frameMultitude;
    public final ImageView imgBluetoothMultitude;
    public final ImageView iplayImgCloseLeft;
    public final ImageView iplayImgCloseRight;
    public final ImageView iplayImgGestureLeft;
    public final ImageView iplayImgGestureRight;
    public final TextView iplayTxtGestureTip;
    public final RightDraggableFrameLayout layoutFragmentPiu;
    private final ConstraintLayout rootView;
    public final TextView titleTxtStatus;
    public final View viewTouch;

    private ActivityPlaySoloBinding(ConstraintLayout constraintLayout, RightDraggableFrameLayout rightDraggableFrameLayout, FrameLayout frameLayout, FrameLayout frameLayout2, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, TextView textView, RightDraggableFrameLayout rightDraggableFrameLayout2, TextView textView2, View view) {
        this.rootView = constraintLayout;
        this.draglayoutDeviceList = rightDraggableFrameLayout;
        this.frameLayout = frameLayout;
        this.frameMultitude = frameLayout2;
        this.imgBluetoothMultitude = imageView;
        this.iplayImgCloseLeft = imageView2;
        this.iplayImgCloseRight = imageView3;
        this.iplayImgGestureLeft = imageView4;
        this.iplayImgGestureRight = imageView5;
        this.iplayTxtGestureTip = textView;
        this.layoutFragmentPiu = rightDraggableFrameLayout2;
        this.titleTxtStatus = textView2;
        this.viewTouch = view;
    }

    public static ActivityPlaySoloBinding bind(View view) {
        View findChildViewById;
        int i = R.id.draglayout_device_list;
        RightDraggableFrameLayout rightDraggableFrameLayout = (RightDraggableFrameLayout) ViewBindings.findChildViewById(view, i);
        if (rightDraggableFrameLayout != null) {
            i = R.id.frame_layout;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.frame_multitude;
                FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout2 != null) {
                    i = R.id.img_bluetooth_multitude;
                    ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView != null) {
                        i = R.id.iplay_img_close_left;
                        ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView2 != null) {
                            i = R.id.iplay_img_close_right;
                            ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView3 != null) {
                                i = R.id.iplay_img_gesture_left;
                                ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView4 != null) {
                                    i = R.id.iplay_img_gesture_right;
                                    ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView5 != null) {
                                        i = R.id.iplay_txt_gesture_tip;
                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView != null) {
                                            i = R.id.layout_fragment_piu;
                                            RightDraggableFrameLayout rightDraggableFrameLayout2 = (RightDraggableFrameLayout) ViewBindings.findChildViewById(view, i);
                                            if (rightDraggableFrameLayout2 != null) {
                                                i = R.id.title_txt_status;
                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_touch))) != null) {
                                                    return new ActivityPlaySoloBinding((ConstraintLayout) view, rightDraggableFrameLayout, frameLayout, frameLayout2, imageView, imageView2, imageView3, imageView4, imageView5, textView, rightDraggableFrameLayout2, textView2, findChildViewById);
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

    public static ActivityPlaySoloBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlaySoloBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_solo, viewGroup, false);
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
