package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogDrawerPlaySelecterBinding implements ViewBinding {
    public final LinearLayout layoutOne;
    public final LinearLayout layoutTwo;
    public final ConstraintLayout playImgBg;
    public final ImageView playImgClose;
    public final ImageView playImgIPlay;
    public final ImageView playImgLinePlay;
    public final ImageView playImgModePlay;
    public final ImageView playImgRemotePlay;
    public final TextView playTxtIPlay;
    public final TextView playTxtLinePlay;
    public final TextView playTxtModePlay;
    public final TextView playTxtRemotePlay;
    public final RecyclerView recyclerView;
    private final ConstraintLayout rootView;

    private DialogDrawerPlaySelecterBinding(ConstraintLayout constraintLayout, LinearLayout linearLayout, LinearLayout linearLayout2, ConstraintLayout constraintLayout2, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, TextView textView, TextView textView2, TextView textView3, TextView textView4, RecyclerView recyclerView) {
        this.rootView = constraintLayout;
        this.layoutOne = linearLayout;
        this.layoutTwo = linearLayout2;
        this.playImgBg = constraintLayout2;
        this.playImgClose = imageView;
        this.playImgIPlay = imageView2;
        this.playImgLinePlay = imageView3;
        this.playImgModePlay = imageView4;
        this.playImgRemotePlay = imageView5;
        this.playTxtIPlay = textView;
        this.playTxtLinePlay = textView2;
        this.playTxtModePlay = textView3;
        this.playTxtRemotePlay = textView4;
        this.recyclerView = recyclerView;
    }

    public static DialogDrawerPlaySelecterBinding bind(View view) {
        int i = R.id.layout_one;
        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
        if (linearLayout != null) {
            i = R.id.layout_two;
            LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
            if (linearLayout2 != null) {
                ConstraintLayout constraintLayout = (ConstraintLayout) view;
                i = R.id.play_img_close;
                ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView != null) {
                    i = R.id.play_img_iPlay;
                    ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView2 != null) {
                        i = R.id.play_img_linePlay;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.play_img_modePlay;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.play_img_remotePlay;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.play_txt_iPlay;
                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView != null) {
                                        i = R.id.play_txt_linePlay;
                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView2 != null) {
                                            i = R.id.play_txt_modePlay;
                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView3 != null) {
                                                i = R.id.play_txt_remotePlay;
                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView4 != null) {
                                                    i = R.id.recycler_view;
                                                    RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                    if (recyclerView != null) {
                                                        return new DialogDrawerPlaySelecterBinding(constraintLayout, linearLayout, linearLayout2, constraintLayout, imageView, imageView2, imageView3, imageView4, imageView5, textView, textView2, textView3, textView4, recyclerView);
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

    public static DialogDrawerPlaySelecterBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogDrawerPlaySelecterBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_drawer_play_selecter, viewGroup, false);
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
