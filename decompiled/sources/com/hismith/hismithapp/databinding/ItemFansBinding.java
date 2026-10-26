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
public final class ItemFansBinding implements ViewBinding {
    public final ImageView fansImgGender;
    public final CircleImageView fansImgHead;
    public final ImageView fansImgProgress;
    public final FrameLayout fansLayoutFollow;
    public final ConstraintLayout fansLayoutUser;
    public final TextView fansTxtFollow;
    public final TextView fansTxtId;
    public final TextView fansTxtName;
    public final TextView fansTxtRemove;
    private final LinearLayout rootView;

    private ItemFansBinding(LinearLayout linearLayout, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, FrameLayout frameLayout, ConstraintLayout constraintLayout, TextView textView, TextView textView2, TextView textView3, TextView textView4) {
        this.rootView = linearLayout;
        this.fansImgGender = imageView;
        this.fansImgHead = circleImageView;
        this.fansImgProgress = imageView2;
        this.fansLayoutFollow = frameLayout;
        this.fansLayoutUser = constraintLayout;
        this.fansTxtFollow = textView;
        this.fansTxtId = textView2;
        this.fansTxtName = textView3;
        this.fansTxtRemove = textView4;
    }

    public static ItemFansBinding bind(View view) {
        int i = R.id.fans_img_gender;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.fans_img_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.fans_img_progress;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.fans_layout_follow;
                    FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                    if (frameLayout != null) {
                        i = R.id.fans_layout_user;
                        ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                        if (constraintLayout != null) {
                            i = R.id.fans_txt_follow;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.fans_txt_id;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    i = R.id.fans_txt_name;
                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView3 != null) {
                                        i = R.id.fans_txt_remove;
                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView4 != null) {
                                            return new ItemFansBinding((LinearLayout) view, imageView, circleImageView, imageView2, frameLayout, constraintLayout, textView, textView2, textView3, textView4);
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

    public static ItemFansBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemFansBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_fans, viewGroup, false);
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
