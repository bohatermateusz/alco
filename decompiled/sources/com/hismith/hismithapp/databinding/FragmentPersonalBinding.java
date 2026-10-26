package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class FragmentPersonalBinding implements ViewBinding {
    public final LinearLayout layoutHead;
    public final ImageView personalImgArrow;
    public final ImageView personalImgGender;
    public final CircleImageView personalImgHead;
    public final ImageView personalImgScan;
    public final ImageView personalImgSetting;
    public final LinearLayout personalLayoutFans;
    public final LinearLayout personalLayoutFollow;
    public final LinearLayout personalLayoutFriend;
    public final LinearLayout personalLayoutInfo;
    public final RecyclerView personalRecycler;
    public final TextView personalTxtConstellation;
    public final TextView personalTxtFans;
    public final TextView personalTxtFansCount;
    public final TextView personalTxtFollow;
    public final TextView personalTxtFollowCount;
    public final TextView personalTxtFriend;
    public final TextView personalTxtFriendCount;
    public final TextView personalTxtName;
    private final LinearLayout rootView;

    private FragmentPersonalBinding(LinearLayout linearLayout, LinearLayout linearLayout2, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, ImageView imageView3, ImageView imageView4, LinearLayout linearLayout3, LinearLayout linearLayout4, LinearLayout linearLayout5, LinearLayout linearLayout6, RecyclerView recyclerView, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8) {
        this.rootView = linearLayout;
        this.layoutHead = linearLayout2;
        this.personalImgArrow = imageView;
        this.personalImgGender = imageView2;
        this.personalImgHead = circleImageView;
        this.personalImgScan = imageView3;
        this.personalImgSetting = imageView4;
        this.personalLayoutFans = linearLayout3;
        this.personalLayoutFollow = linearLayout4;
        this.personalLayoutFriend = linearLayout5;
        this.personalLayoutInfo = linearLayout6;
        this.personalRecycler = recyclerView;
        this.personalTxtConstellation = textView;
        this.personalTxtFans = textView2;
        this.personalTxtFansCount = textView3;
        this.personalTxtFollow = textView4;
        this.personalTxtFollowCount = textView5;
        this.personalTxtFriend = textView6;
        this.personalTxtFriendCount = textView7;
        this.personalTxtName = textView8;
    }

    public static FragmentPersonalBinding bind(View view) {
        int i = R.id.layout_head;
        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
        if (linearLayout != null) {
            i = R.id.personal_img_arrow;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.personal_img_gender;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.personal_img_head;
                    CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                    if (circleImageView != null) {
                        i = R.id.personal_img_scan;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.personal_img_setting;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.personal_layout_fans;
                                LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout2 != null) {
                                    i = R.id.personal_layout_follow;
                                    LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout3 != null) {
                                        i = R.id.personal_layout_friend;
                                        LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout4 != null) {
                                            i = R.id.personal_layout_info;
                                            LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout5 != null) {
                                                i = R.id.personal_recycler;
                                                RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                if (recyclerView != null) {
                                                    i = R.id.personal_txt_constellation;
                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView != null) {
                                                        i = R.id.personal_txt_fans;
                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView2 != null) {
                                                            i = R.id.personal_txt_fansCount;
                                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView3 != null) {
                                                                i = R.id.personal_txt_follow;
                                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView4 != null) {
                                                                    i = R.id.personal_txt_followCount;
                                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView5 != null) {
                                                                        i = R.id.personal_txt_friend;
                                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView6 != null) {
                                                                            i = R.id.personal_txt_friendCount;
                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView7 != null) {
                                                                                i = R.id.personal_txt_name;
                                                                                TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView8 != null) {
                                                                                    return new FragmentPersonalBinding((LinearLayout) view, linearLayout, imageView, imageView2, circleImageView, imageView3, imageView4, linearLayout2, linearLayout3, linearLayout4, linearLayout5, recyclerView, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8);
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

    public static FragmentPersonalBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentPersonalBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal, viewGroup, false);
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
