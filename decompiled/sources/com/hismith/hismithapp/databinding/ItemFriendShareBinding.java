package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemFriendShareBinding implements ViewBinding {
    public final ImageView itemShareFriendImgGender;
    public final CircleImageView itemShareFriendImgHead;
    public final FrameLayout itemShareFriendLayoutSelect;
    public final TextView itemShareFriendTxtName;
    private final FrameLayout rootView;

    private ItemFriendShareBinding(FrameLayout frameLayout, ImageView imageView, CircleImageView circleImageView, FrameLayout frameLayout2, TextView textView) {
        this.rootView = frameLayout;
        this.itemShareFriendImgGender = imageView;
        this.itemShareFriendImgHead = circleImageView;
        this.itemShareFriendLayoutSelect = frameLayout2;
        this.itemShareFriendTxtName = textView;
    }

    public static ItemFriendShareBinding bind(View view) {
        int i = R.id.item_share_friend_img_gender;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_share_friend_img_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.item_share_friend_layout_select;
                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                if (frameLayout != null) {
                    i = R.id.item_share_friend_txt_name;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        return new ItemFriendShareBinding((FrameLayout) view, imageView, circleImageView, frameLayout, textView);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ItemFriendShareBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemFriendShareBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_friend_share, viewGroup, false);
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
