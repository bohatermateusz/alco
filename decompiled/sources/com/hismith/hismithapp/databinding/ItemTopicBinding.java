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
public final class ItemTopicBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final GridLayout topicGridLayout;
    public final ImageView topicImgDown;
    public final ImageView topicImgFavorite;
    public final CircleImageView topicImgHead;
    public final ImageView topicImgLike;
    public final ImageView topicImgProgressFavorite;
    public final ImageView topicImgProgressLike;
    public final ImageView topicImgRead;
    public final ImageView topicImgReview;
    public final LinearLayout topicLayoutFavorite;
    public final LinearLayout topicLayoutItem;
    public final LinearLayout topicLayoutLike;
    public final TextView topicTxtContent;
    public final TextView topicTxtFavorite;
    public final TextView topicTxtLike;
    public final TextView topicTxtName;
    public final TextView topicTxtRead;
    public final TextView topicTxtReview;
    public final TextView topicTxtTime;
    public final TextView topicTxtTitle;

    private ItemTopicBinding(LinearLayout linearLayout, GridLayout gridLayout, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8) {
        this.rootView = linearLayout;
        this.topicGridLayout = gridLayout;
        this.topicImgDown = imageView;
        this.topicImgFavorite = imageView2;
        this.topicImgHead = circleImageView;
        this.topicImgLike = imageView3;
        this.topicImgProgressFavorite = imageView4;
        this.topicImgProgressLike = imageView5;
        this.topicImgRead = imageView6;
        this.topicImgReview = imageView7;
        this.topicLayoutFavorite = linearLayout2;
        this.topicLayoutItem = linearLayout3;
        this.topicLayoutLike = linearLayout4;
        this.topicTxtContent = textView;
        this.topicTxtFavorite = textView2;
        this.topicTxtLike = textView3;
        this.topicTxtName = textView4;
        this.topicTxtRead = textView5;
        this.topicTxtReview = textView6;
        this.topicTxtTime = textView7;
        this.topicTxtTitle = textView8;
    }

    public static ItemTopicBinding bind(View view) {
        int i = R.id.topic_grid_layout;
        GridLayout gridLayout = (GridLayout) ViewBindings.findChildViewById(view, i);
        if (gridLayout != null) {
            i = R.id.topic_img_down;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.topic_img_favorite;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.topic_img_head;
                    CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                    if (circleImageView != null) {
                        i = R.id.topic_img_like;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.topic_img_progress_favorite;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.topic_img_progress_like;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.topic_img_read;
                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView6 != null) {
                                        i = R.id.topic_img_review;
                                        ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView7 != null) {
                                            i = R.id.topic_layout_favorite;
                                            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout != null) {
                                                LinearLayout linearLayout2 = (LinearLayout) view;
                                                i = R.id.topic_layout_like;
                                                LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                if (linearLayout3 != null) {
                                                    i = R.id.topic_txt_content;
                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView != null) {
                                                        i = R.id.topic_txt_favorite;
                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView2 != null) {
                                                            i = R.id.topic_txt_like;
                                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView3 != null) {
                                                                i = R.id.topic_txt_name;
                                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView4 != null) {
                                                                    i = R.id.topic_txt_read;
                                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView5 != null) {
                                                                        i = R.id.topic_txt_review;
                                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView6 != null) {
                                                                            i = R.id.topic_txt_time;
                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView7 != null) {
                                                                                i = R.id.topic_txt_title;
                                                                                TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView8 != null) {
                                                                                    return new ItemTopicBinding(linearLayout2, gridLayout, imageView, imageView2, circleImageView, imageView3, imageView4, imageView5, imageView6, imageView7, linearLayout, linearLayout2, linearLayout3, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8);
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

    public static ItemTopicBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemTopicBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_topic, viewGroup, false);
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
