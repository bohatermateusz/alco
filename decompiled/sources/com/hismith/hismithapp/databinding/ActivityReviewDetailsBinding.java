package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.widget.NestedScrollView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityReviewDetailsBinding implements ViewBinding {
    public final FrameLayout layoutHighlight;
    public final EditText reviewDetailsEditReview;
    public final ImageView reviewDetailsImgDelete;
    public final CircleImageView reviewDetailsImgHead;
    public final ImageView reviewDetailsImgLike;
    public final ConstraintLayout reviewDetailsLayoutLike;
    public final LinearLayout reviewDetailsLayoutLikeCount;
    public final RecyclerView reviewDetailsRecycler;
    public final SmartRefreshLayout reviewDetailsRefreshLayout;
    public final TextView reviewDetailsTxtContent;
    public final TextView reviewDetailsTxtLikeCount;
    public final TextView reviewDetailsTxtLikeCount1;
    public final TextView reviewDetailsTxtName;
    public final TextView reviewDetailsTxtPublish;
    public final TextView reviewDetailsTxtReport;
    public final TextView reviewDetailsTxtTime;
    public final TextView reviewDetailsTxtTopicTitle;
    private final LinearLayout rootView;
    public final NestedScrollView scrollView;
    public final TextView titleTxtName;

    private ActivityReviewDetailsBinding(LinearLayout linearLayout, FrameLayout frameLayout, EditText editText, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, ConstraintLayout constraintLayout, LinearLayout linearLayout2, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, NestedScrollView nestedScrollView, TextView textView9) {
        this.rootView = linearLayout;
        this.layoutHighlight = frameLayout;
        this.reviewDetailsEditReview = editText;
        this.reviewDetailsImgDelete = imageView;
        this.reviewDetailsImgHead = circleImageView;
        this.reviewDetailsImgLike = imageView2;
        this.reviewDetailsLayoutLike = constraintLayout;
        this.reviewDetailsLayoutLikeCount = linearLayout2;
        this.reviewDetailsRecycler = recyclerView;
        this.reviewDetailsRefreshLayout = smartRefreshLayout;
        this.reviewDetailsTxtContent = textView;
        this.reviewDetailsTxtLikeCount = textView2;
        this.reviewDetailsTxtLikeCount1 = textView3;
        this.reviewDetailsTxtName = textView4;
        this.reviewDetailsTxtPublish = textView5;
        this.reviewDetailsTxtReport = textView6;
        this.reviewDetailsTxtTime = textView7;
        this.reviewDetailsTxtTopicTitle = textView8;
        this.scrollView = nestedScrollView;
        this.titleTxtName = textView9;
    }

    public static ActivityReviewDetailsBinding bind(View view) {
        int i = R.id.layout_highlight;
        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
        if (frameLayout != null) {
            i = R.id.review_details_edit_review;
            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText != null) {
                i = R.id.review_details_img_delete;
                ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView != null) {
                    i = R.id.review_details_img_head;
                    CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                    if (circleImageView != null) {
                        i = R.id.review_details_img_like;
                        ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView2 != null) {
                            i = R.id.review_details_layout_like;
                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout != null) {
                                i = R.id.review_details_layout_likeCount;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.review_details_recycler;
                                    RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                    if (recyclerView != null) {
                                        i = R.id.review_details_refreshLayout;
                                        SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                                        if (smartRefreshLayout != null) {
                                            i = R.id.review_details_txt_content;
                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView != null) {
                                                i = R.id.review_details_txt_likeCount;
                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView2 != null) {
                                                    i = R.id.review_details_txt_likeCount_1;
                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView3 != null) {
                                                        i = R.id.review_details_txt_name;
                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView4 != null) {
                                                            i = R.id.review_details_txt_publish;
                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView5 != null) {
                                                                i = R.id.review_details_txt_report;
                                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView6 != null) {
                                                                    i = R.id.review_details_txt_time;
                                                                    TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView7 != null) {
                                                                        i = R.id.review_details_txt_topic_title;
                                                                        TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView8 != null) {
                                                                            i = R.id.scroll_view;
                                                                            NestedScrollView nestedScrollView = (NestedScrollView) ViewBindings.findChildViewById(view, i);
                                                                            if (nestedScrollView != null) {
                                                                                i = R.id.title_txt_name;
                                                                                TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView9 != null) {
                                                                                    return new ActivityReviewDetailsBinding((LinearLayout) view, frameLayout, editText, imageView, circleImageView, imageView2, constraintLayout, linearLayout, recyclerView, smartRefreshLayout, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, nestedScrollView, textView9);
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

    public static ActivityReviewDetailsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityReviewDetailsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_review_details, viewGroup, false);
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
