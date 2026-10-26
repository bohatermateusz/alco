package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.GridLayout;

/* loaded from: classes3.dex */
public final class ItemDetailsTopicBinding implements ViewBinding {
    public final GridLayout detailsTopicGridLayout;
    public final ImageView detailsTopicImgFavorite;
    public final CircleImageView detailsTopicImgHead;
    public final ImageView detailsTopicImgLike;
    public final ImageView detailsTopicImgProgressFavorite;
    public final ImageView detailsTopicImgProgressLike;
    public final ImageView detailsTopicImgRead;
    public final ImageView detailsTopicImgReview;
    public final LinearLayout detailsTopicLayoutFavorite;
    public final LinearLayout detailsTopicLayoutItem;
    public final LinearLayout detailsTopicLayoutLike;
    public final TextView detailsTopicTxtContent;
    public final TextView detailsTopicTxtDelete;
    public final TextView detailsTopicTxtFavorite;
    public final TextView detailsTopicTxtLike;
    public final TextView detailsTopicTxtName;
    public final TextView detailsTopicTxtRead;
    public final TextView detailsTopicTxtReview;
    public final TextView detailsTopicTxtTime;
    public final TextView detailsTopicTxtTitle;
    private final LinearLayout rootView;

    private ItemDetailsTopicBinding(LinearLayout linearLayout, GridLayout gridLayout, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9) {
        this.rootView = linearLayout;
        this.detailsTopicGridLayout = gridLayout;
        this.detailsTopicImgFavorite = imageView;
        this.detailsTopicImgHead = circleImageView;
        this.detailsTopicImgLike = imageView2;
        this.detailsTopicImgProgressFavorite = imageView3;
        this.detailsTopicImgProgressLike = imageView4;
        this.detailsTopicImgRead = imageView5;
        this.detailsTopicImgReview = imageView6;
        this.detailsTopicLayoutFavorite = linearLayout2;
        this.detailsTopicLayoutItem = linearLayout3;
        this.detailsTopicLayoutLike = linearLayout4;
        this.detailsTopicTxtContent = textView;
        this.detailsTopicTxtDelete = textView2;
        this.detailsTopicTxtFavorite = textView3;
        this.detailsTopicTxtLike = textView4;
        this.detailsTopicTxtName = textView5;
        this.detailsTopicTxtRead = textView6;
        this.detailsTopicTxtReview = textView7;
        this.detailsTopicTxtTime = textView8;
        this.detailsTopicTxtTitle = textView9;
    }

    public static ItemDetailsTopicBinding bind(View view) {
        int i = R.id.details_topic_grid_layout;
        GridLayout gridLayout = (GridLayout) ViewBindings.findChildViewById(view, i);
        if (gridLayout != null) {
            i = R.id.details_topic_img_favorite;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.details_topic_img_head;
                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                if (circleImageView != null) {
                    i = R.id.details_topic_img_like;
                    ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView2 != null) {
                        i = R.id.details_topic_img_progress_favorite;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.details_topic_img_progress_like;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.details_topic_img_read;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.details_topic_img_review;
                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView6 != null) {
                                        i = R.id.details_topic_layout_favorite;
                                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout != null) {
                                            LinearLayout linearLayout2 = (LinearLayout) view;
                                            i = R.id.details_topic_layout_like;
                                            LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout3 != null) {
                                                i = R.id.details_topic_txt_content;
                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView != null) {
                                                    i = R.id.details_topic_txt_delete;
                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView2 != null) {
                                                        i = R.id.details_topic_txt_favorite;
                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView3 != null) {
                                                            i = R.id.details_topic_txt_like;
                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView4 != null) {
                                                                i = R.id.details_topic_txt_name;
                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView5 != null) {
                                                                    i = R.id.details_topic_txt_read;
                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView6 != null) {
                                                                        i = R.id.details_topic_txt_review;
                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView7 != null) {
                                                                            i = R.id.details_topic_txt_time;
                                                                            TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView8 != null) {
                                                                                i = R.id.details_topic_txt_title;
                                                                                TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView9 != null) {
                                                                                    return new ItemDetailsTopicBinding(linearLayout2, gridLayout, imageView, circleImageView, imageView2, imageView3, imageView4, imageView5, imageView6, linearLayout, linearLayout2, linearLayout3, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9);
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

    public static ItemDetailsTopicBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemDetailsTopicBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_details_topic, viewGroup, false);
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
