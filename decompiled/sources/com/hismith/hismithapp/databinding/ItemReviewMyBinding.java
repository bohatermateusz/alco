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
public final class ItemReviewMyBinding implements ViewBinding {
    public final GridLayout reviewGridLayout;
    public final ImageView reviewImgCollect;
    public final ImageView reviewImgDown;
    public final ImageView reviewImgHead;
    public final ImageView reviewImgKing;
    public final ImageView reviewImgLike;
    public final CircleImageView reviewImgMyHead;
    public final ImageView reviewImgProgressCollect;
    public final ImageView reviewImgProgressLike;
    public final ImageView reviewImgRead;
    public final ImageView reviewImgReview;
    public final LinearLayout reviewLayoutAction;
    public final LinearLayout reviewLayoutCollect;
    public final LinearLayout reviewLayoutLike;
    public final LinearLayout reviewLayoutRead;
    public final LinearLayout reviewLayoutReview;
    public final LinearLayout reviewLayoutTopic;
    public final LinearLayout reviewLayoutUser;
    public final TextView reviewTxtCollect;
    public final TextView reviewTxtContent;
    public final TextView reviewTxtLike;
    public final TextView reviewTxtMyContent;
    public final TextView reviewTxtMyName;
    public final TextView reviewTxtMyTime;
    public final TextView reviewTxtName;
    public final TextView reviewTxtRead;
    public final TextView reviewTxtReview;
    public final TextView reviewTxtTitle;
    private final LinearLayout rootView;

    private ItemReviewMyBinding(LinearLayout linearLayout, GridLayout gridLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, CircleImageView circleImageView, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, LinearLayout linearLayout5, LinearLayout linearLayout6, LinearLayout linearLayout7, LinearLayout linearLayout8, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10) {
        this.rootView = linearLayout;
        this.reviewGridLayout = gridLayout;
        this.reviewImgCollect = imageView;
        this.reviewImgDown = imageView2;
        this.reviewImgHead = imageView3;
        this.reviewImgKing = imageView4;
        this.reviewImgLike = imageView5;
        this.reviewImgMyHead = circleImageView;
        this.reviewImgProgressCollect = imageView6;
        this.reviewImgProgressLike = imageView7;
        this.reviewImgRead = imageView8;
        this.reviewImgReview = imageView9;
        this.reviewLayoutAction = linearLayout2;
        this.reviewLayoutCollect = linearLayout3;
        this.reviewLayoutLike = linearLayout4;
        this.reviewLayoutRead = linearLayout5;
        this.reviewLayoutReview = linearLayout6;
        this.reviewLayoutTopic = linearLayout7;
        this.reviewLayoutUser = linearLayout8;
        this.reviewTxtCollect = textView;
        this.reviewTxtContent = textView2;
        this.reviewTxtLike = textView3;
        this.reviewTxtMyContent = textView4;
        this.reviewTxtMyName = textView5;
        this.reviewTxtMyTime = textView6;
        this.reviewTxtName = textView7;
        this.reviewTxtRead = textView8;
        this.reviewTxtReview = textView9;
        this.reviewTxtTitle = textView10;
    }

    public static ItemReviewMyBinding bind(View view) {
        int i = R.id.review_grid_layout;
        GridLayout gridLayout = (GridLayout) ViewBindings.findChildViewById(view, i);
        if (gridLayout != null) {
            i = R.id.review_img_collect;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.review_img_down;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.review_img_head;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.review_img_king;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.review_img_like;
                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView5 != null) {
                                i = R.id.review_img_my_head;
                                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                                if (circleImageView != null) {
                                    i = R.id.review_img_progress_collect;
                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView6 != null) {
                                        i = R.id.review_img_progress_like;
                                        ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView7 != null) {
                                            i = R.id.review_img_read;
                                            ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView8 != null) {
                                                i = R.id.review_img_review;
                                                ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView9 != null) {
                                                    i = R.id.review_layout_action;
                                                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                    if (linearLayout != null) {
                                                        i = R.id.review_layout_collect;
                                                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                        if (linearLayout2 != null) {
                                                            i = R.id.review_layout_like;
                                                            LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                            if (linearLayout3 != null) {
                                                                i = R.id.review_layout_read;
                                                                LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                if (linearLayout4 != null) {
                                                                    i = R.id.review_layout_review;
                                                                    LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (linearLayout5 != null) {
                                                                        i = R.id.review_layout_topic;
                                                                        LinearLayout linearLayout6 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (linearLayout6 != null) {
                                                                            i = R.id.review_layout_user;
                                                                            LinearLayout linearLayout7 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                            if (linearLayout7 != null) {
                                                                                i = R.id.review_txt_collect;
                                                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView != null) {
                                                                                    i = R.id.review_txt_content;
                                                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView2 != null) {
                                                                                        i = R.id.review_txt_like;
                                                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView3 != null) {
                                                                                            i = R.id.review_txt_my_content;
                                                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView4 != null) {
                                                                                                i = R.id.review_txt_my_name;
                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView5 != null) {
                                                                                                    i = R.id.review_txt_my_time;
                                                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView6 != null) {
                                                                                                        i = R.id.review_txt_name;
                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView7 != null) {
                                                                                                            i = R.id.review_txt_read;
                                                                                                            TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (textView8 != null) {
                                                                                                                i = R.id.review_txt_review;
                                                                                                                TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView9 != null) {
                                                                                                                    i = R.id.review_txt_title;
                                                                                                                    TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (textView10 != null) {
                                                                                                                        return new ItemReviewMyBinding((LinearLayout) view, gridLayout, imageView, imageView2, imageView3, imageView4, imageView5, circleImageView, imageView6, imageView7, imageView8, imageView9, linearLayout, linearLayout2, linearLayout3, linearLayout4, linearLayout5, linearLayout6, linearLayout7, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9, textView10);
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

    public static ItemReviewMyBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemReviewMyBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_review_my, viewGroup, false);
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
