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
public final class ItemReviewDetailsBinding implements ViewBinding {
    public final ImageView itemReviewDetailsImgDelete;
    public final CircleImageView itemReviewDetailsImgHead;
    public final ImageView itemReviewDetailsImgLike;
    public final TextView itemReviewDetailsTxtContent;
    public final TextView itemReviewDetailsTxtLike;
    public final TextView itemReviewDetailsTxtName;
    public final TextView itemReviewDetailsTxtReply;
    public final TextView itemReviewDetailsTxtTime;
    private final LinearLayout rootView;

    private ItemReviewDetailsBinding(LinearLayout linearLayout, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = linearLayout;
        this.itemReviewDetailsImgDelete = imageView;
        this.itemReviewDetailsImgHead = circleImageView;
        this.itemReviewDetailsImgLike = imageView2;
        this.itemReviewDetailsTxtContent = textView;
        this.itemReviewDetailsTxtLike = textView2;
        this.itemReviewDetailsTxtName = textView3;
        this.itemReviewDetailsTxtReply = textView4;
        this.itemReviewDetailsTxtTime = textView5;
    }

    public static ItemReviewDetailsBinding bind(View view) {
        int i = R.id.item_review_details_img_delete;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_review_details_img_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.item_review_details_img_like;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.item_review_details_txt_content;
                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView != null) {
                        i = R.id.item_review_details_txt_like;
                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView2 != null) {
                            i = R.id.item_review_details_txt_name;
                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView3 != null) {
                                i = R.id.item_review_details_txt_reply;
                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView4 != null) {
                                    i = R.id.item_review_details_txt_time;
                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView5 != null) {
                                        return new ItemReviewDetailsBinding((LinearLayout) view, imageView, circleImageView, imageView2, textView, textView2, textView3, textView4, textView5);
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

    public static ItemReviewDetailsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemReviewDetailsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_review_details, viewGroup, false);
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
