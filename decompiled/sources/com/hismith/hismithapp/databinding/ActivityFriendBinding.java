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
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityFriendBinding implements ViewBinding {
    public final FrameLayout friendLayout;
    public final SlideRecyclerView friendRecyclerView;
    public final SmartRefreshLayout friendRefreshLayout;
    private final LinearLayout rootView;
    public final ImageView titleImgAdd;
    public final LinearLayout titleLayout;
    public final TextView titleTxtName;

    private ActivityFriendBinding(LinearLayout linearLayout, FrameLayout frameLayout, SlideRecyclerView slideRecyclerView, SmartRefreshLayout smartRefreshLayout, ImageView imageView, LinearLayout linearLayout2, TextView textView) {
        this.rootView = linearLayout;
        this.friendLayout = frameLayout;
        this.friendRecyclerView = slideRecyclerView;
        this.friendRefreshLayout = smartRefreshLayout;
        this.titleImgAdd = imageView;
        this.titleLayout = linearLayout2;
        this.titleTxtName = textView;
    }

    public static ActivityFriendBinding bind(View view) {
        int i = R.id.friend_layout;
        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
        if (frameLayout != null) {
            i = R.id.friend_recycler_view;
            SlideRecyclerView slideRecyclerView = (SlideRecyclerView) ViewBindings.findChildViewById(view, i);
            if (slideRecyclerView != null) {
                i = R.id.friend_refreshLayout;
                SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                if (smartRefreshLayout != null) {
                    i = R.id.title_img_add;
                    ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView != null) {
                        i = R.id.title_layout;
                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout != null) {
                            i = R.id.title_txt_name;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                return new ActivityFriendBinding((LinearLayout) view, frameLayout, slideRecyclerView, smartRefreshLayout, imageView, linearLayout, textView);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityFriendBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityFriendBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_friend, viewGroup, false);
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
