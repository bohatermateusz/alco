package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class DialogLineUserInfoBinding implements ViewBinding {
    public final CircleImageView dialogLineInfoImgHead;
    public final FrameLayout dialogLineInfoLayoutCaptain;
    public final ProgressBar dialogLineInfoProgressCaptain;
    public final TextView dialogLineInfoTxtCaptain;
    public final TextView dialogLineInfoTxtFans;
    public final TextView dialogLineInfoTxtFansCount;
    public final TextView dialogLineInfoTxtFollow;
    public final TextView dialogLineInfoTxtFollowCount;
    public final TextView dialogLineInfoTxtName;
    private final RelativeLayout rootView;

    private DialogLineUserInfoBinding(RelativeLayout relativeLayout, CircleImageView circleImageView, FrameLayout frameLayout, ProgressBar progressBar, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6) {
        this.rootView = relativeLayout;
        this.dialogLineInfoImgHead = circleImageView;
        this.dialogLineInfoLayoutCaptain = frameLayout;
        this.dialogLineInfoProgressCaptain = progressBar;
        this.dialogLineInfoTxtCaptain = textView;
        this.dialogLineInfoTxtFans = textView2;
        this.dialogLineInfoTxtFansCount = textView3;
        this.dialogLineInfoTxtFollow = textView4;
        this.dialogLineInfoTxtFollowCount = textView5;
        this.dialogLineInfoTxtName = textView6;
    }

    public static DialogLineUserInfoBinding bind(View view) {
        int i = R.id.dialog_line_info_img_head;
        CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
        if (circleImageView != null) {
            i = R.id.dialog_line_info_layout_captain;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.dialog_line_info_progress_captain;
                ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                if (progressBar != null) {
                    i = R.id.dialog_line_info_txt_captain;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.dialog_line_info_txt_fans;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            i = R.id.dialog_line_info_txt_fans_count;
                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView3 != null) {
                                i = R.id.dialog_line_info_txt_follow;
                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView4 != null) {
                                    i = R.id.dialog_line_info_txt_follow_count;
                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView5 != null) {
                                        i = R.id.dialog_line_info_txt_name;
                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView6 != null) {
                                            return new DialogLineUserInfoBinding((RelativeLayout) view, circleImageView, frameLayout, progressBar, textView, textView2, textView3, textView4, textView5, textView6);
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

    public static DialogLineUserInfoBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogLineUserInfoBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_line_user_info, viewGroup, false);
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
