package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemLineUserBinding implements ViewBinding {
    public final ImageView itemLineImgProgress;
    public final ImageView itemLineImgRemoveUser;
    public final ImageView itemLineImgUserCaptain;
    public final ImageView itemLineImgUserCrown;
    public final ImageView itemLineImgUserGender;
    public final CircleImageView itemLineImgUserHead;
    public final FrameLayout itemLineLayoutRemoveUser;
    public final FrameLayout itemLineLayoutUserFollow;
    public final TextView itemLineTxtUserFollow;
    public final TextView itemLineTxtUserName;
    private final LinearLayout rootView;

    private ItemLineUserBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, CircleImageView circleImageView, FrameLayout frameLayout, FrameLayout frameLayout2, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.itemLineImgProgress = imageView;
        this.itemLineImgRemoveUser = imageView2;
        this.itemLineImgUserCaptain = imageView3;
        this.itemLineImgUserCrown = imageView4;
        this.itemLineImgUserGender = imageView5;
        this.itemLineImgUserHead = circleImageView;
        this.itemLineLayoutRemoveUser = frameLayout;
        this.itemLineLayoutUserFollow = frameLayout2;
        this.itemLineTxtUserFollow = textView;
        this.itemLineTxtUserName = textView2;
    }

    public static ItemLineUserBinding bind(View view) {
        int i = R.id.item_line_img_progress;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_line_img_remove_user;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.item_line_img_user_captain;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.item_line_img_user_crown;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.item_line_img_user_gender;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.item_line_img_user_head;
                            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                            if (circleImageView != null) {
                                i = R.id.item_line_layout_remove_user;
                                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                if (frameLayout != null) {
                                    i = R.id.item_line_layout_user_follow;
                                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                    if (frameLayout2 != null) {
                                        i = R.id.item_line_txt_user_follow;
                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView != null) {
                                            i = R.id.item_line_txt_user_name;
                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView2 != null) {
                                                return new ItemLineUserBinding((LinearLayout) view, imageView, imageView2, imageView3, imageView4, imageView5, circleImageView, frameLayout, frameLayout2, textView, textView2);
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

    public static ItemLineUserBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemLineUserBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_line_user, viewGroup, false);
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
