package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.widget.NestedScrollView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.RichText;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityTopicContentBinding implements ViewBinding {
    public final EditText contentEditReview;
    public final ImageView contentImgCollect;
    public final ImageView contentImgDown;
    public final CircleImageView contentImgHead;
    public final ImageView contentImgLike;
    public final ImageView contentImgMore;
    public final ImageView contentImgReview;
    public final ImageView contentImgSearch;
    public final ImageView contentImgView;
    public final RecyclerView contentRecycler;
    public final SmartRefreshLayout contentRefreshLayout;
    public final NestedScrollView contentScrollView;
    public final TextView contentTxtCollect;
    public final TextView contentTxtCommentCount;
    public final RichText contentTxtContent;
    public final TextView contentTxtLike;
    public final TextView contentTxtName;
    public final TextView contentTxtPublish;
    public final TextView contentTxtReview;
    public final TextView contentTxtTime;
    public final TextView contentTxtTitle;
    public final TextView contentTxtView;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityTopicContentBinding(LinearLayout linearLayout, EditText editText, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, NestedScrollView nestedScrollView, TextView textView, TextView textView2, RichText richText, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10) {
        this.rootView = linearLayout;
        this.contentEditReview = editText;
        this.contentImgCollect = imageView;
        this.contentImgDown = imageView2;
        this.contentImgHead = circleImageView;
        this.contentImgLike = imageView3;
        this.contentImgMore = imageView4;
        this.contentImgReview = imageView5;
        this.contentImgSearch = imageView6;
        this.contentImgView = imageView7;
        this.contentRecycler = recyclerView;
        this.contentRefreshLayout = smartRefreshLayout;
        this.contentScrollView = nestedScrollView;
        this.contentTxtCollect = textView;
        this.contentTxtCommentCount = textView2;
        this.contentTxtContent = richText;
        this.contentTxtLike = textView3;
        this.contentTxtName = textView4;
        this.contentTxtPublish = textView5;
        this.contentTxtReview = textView6;
        this.contentTxtTime = textView7;
        this.contentTxtTitle = textView8;
        this.contentTxtView = textView9;
        this.titleTxtName = textView10;
    }

    public static ActivityTopicContentBinding bind(View view) {
        int i = R.id.content_edit_review;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.content_img_collect;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.content_img_down;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.content_img_head;
                    CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                    if (circleImageView != null) {
                        i = R.id.content_img_like;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.content_img_more;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.content_img_review;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.content_img_search;
                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView6 != null) {
                                        i = R.id.content_img_view;
                                        ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView7 != null) {
                                            i = R.id.content_recycler;
                                            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                            if (recyclerView != null) {
                                                i = R.id.content_refreshLayout;
                                                SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                                                if (smartRefreshLayout != null) {
                                                    i = R.id.content_scrollView;
                                                    NestedScrollView nestedScrollView = (NestedScrollView) ViewBindings.findChildViewById(view, i);
                                                    if (nestedScrollView != null) {
                                                        i = R.id.content_txt_collect;
                                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView != null) {
                                                            i = R.id.content_txt_comment_count;
                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView2 != null) {
                                                                i = R.id.content_txt_content;
                                                                RichText richText = (RichText) ViewBindings.findChildViewById(view, i);
                                                                if (richText != null) {
                                                                    i = R.id.content_txt_like;
                                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView3 != null) {
                                                                        i = R.id.content_txt_name;
                                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView4 != null) {
                                                                            i = R.id.content_txt_publish;
                                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView5 != null) {
                                                                                i = R.id.content_txt_review;
                                                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView6 != null) {
                                                                                    i = R.id.content_txt_time;
                                                                                    TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView7 != null) {
                                                                                        i = R.id.content_txt_title;
                                                                                        TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView8 != null) {
                                                                                            i = R.id.content_txt_view;
                                                                                            TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView9 != null) {
                                                                                                i = R.id.title_txt_name;
                                                                                                TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView10 != null) {
                                                                                                    return new ActivityTopicContentBinding((LinearLayout) view, editText, imageView, imageView2, circleImageView, imageView3, imageView4, imageView5, imageView6, imageView7, recyclerView, smartRefreshLayout, nestedScrollView, textView, textView2, richText, textView3, textView4, textView5, textView6, textView7, textView8, textView9, textView10);
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

    public static ActivityTopicContentBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityTopicContentBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_topic_content, viewGroup, false);
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
