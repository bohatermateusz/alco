package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemLineRoomBinding implements ViewBinding {
    public final ImageView itemLineRoomImgRoomLock;
    public final CircleImageView itemLineRoomImgUserHead;
    public final LinearLayout itemLineRoomLayout;
    public final TextView itemLineRoomTxtEnter;
    public final TextView itemLineRoomTxtRoomId;
    public final TextView itemLineRoomTxtRoomName;
    public final TextView itemLineRoomTxtUserCount;
    public final TextView itemLineRoomTxtUserName;
    private final RelativeLayout rootView;

    private ItemLineRoomBinding(RelativeLayout relativeLayout, ImageView imageView, CircleImageView circleImageView, LinearLayout linearLayout, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = relativeLayout;
        this.itemLineRoomImgRoomLock = imageView;
        this.itemLineRoomImgUserHead = circleImageView;
        this.itemLineRoomLayout = linearLayout;
        this.itemLineRoomTxtEnter = textView;
        this.itemLineRoomTxtRoomId = textView2;
        this.itemLineRoomTxtRoomName = textView3;
        this.itemLineRoomTxtUserCount = textView4;
        this.itemLineRoomTxtUserName = textView5;
    }

    public static ItemLineRoomBinding bind(View view) {
        int i = R.id.item_line_room_img_room_lock;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_line_room_img_user_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.item_line_room_layout;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.item_line_room_txt_enter;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.item_line_room_txt_room_id;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            i = R.id.item_line_room_txt_room_name;
                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView3 != null) {
                                i = R.id.item_line_room_txt_user_count;
                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView4 != null) {
                                    i = R.id.item_line_room_txt_user_name;
                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView5 != null) {
                                        return new ItemLineRoomBinding((RelativeLayout) view, imageView, circleImageView, linearLayout, textView, textView2, textView3, textView4, textView5);
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

    public static ItemLineRoomBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemLineRoomBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_line_room, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public RelativeLayout getRoot() {
        return this.rootView;
    }
}
