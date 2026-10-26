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

/* loaded from: classes3.dex */
public final class ItemTopicReviewBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final ImageView topicReviewImgDelete;
    public final CircleImageView topicReviewImgHead;
    public final ImageView topicReviewImgLike;
    public final LinearLayout topicReviewLayoutReply;
    public final TextView topicReviewTxtContent;
    public final TextView topicReviewTxtDate;
    public final TextView topicReviewTxtLike;
    public final TextView topicReviewTxtName;
    public final TextView topicReviewTxtReply;
    public final TextView topicReviewTxtReplyLabel;

    private ItemTopicReviewBinding(LinearLayout linearLayout, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, LinearLayout linearLayout2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6) {
        this.rootView = linearLayout;
        this.topicReviewImgDelete = imageView;
        this.topicReviewImgHead = circleImageView;
        this.topicReviewImgLike = imageView2;
        this.topicReviewLayoutReply = linearLayout2;
        this.topicReviewTxtContent = textView;
        this.topicReviewTxtDate = textView2;
        this.topicReviewTxtLike = textView3;
        this.topicReviewTxtName = textView4;
        this.topicReviewTxtReply = textView5;
        this.topicReviewTxtReplyLabel = textView6;
    }

    public static ItemTopicReviewBinding bind(View view) {
        int i = R.id.topic_review_img_delete;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.topic_review_img_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.topic_review_img_like;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.topic_review_layout_reply;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.topic_review_txt_content;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.topic_review_txt_date;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.topic_review_txt_like;
                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView3 != null) {
                                    i = R.id.topic_review_txt_name;
                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView4 != null) {
                                        i = R.id.topic_review_txt_reply;
                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView5 != null) {
                                            i = R.id.topic_review_txt_reply_label;
                                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView6 != null) {
                                                return new ItemTopicReviewBinding((LinearLayout) view, imageView, circleImageView, imageView2, linearLayout, textView, textView2, textView3, textView4, textView5, textView6);
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

    public static ItemTopicReviewBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemTopicReviewBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_topic_review, viewGroup, false);
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
