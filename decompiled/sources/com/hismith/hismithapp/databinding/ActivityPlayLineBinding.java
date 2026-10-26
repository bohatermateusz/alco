package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityPlayLineBinding implements ViewBinding {
    public final ImageView imgPeoplesBg;
    public final EditText playLineEditMessage;
    public final ImageView playLineImgPeoples;
    public final LinearLayout playLineLayoutLoading;
    public final RecyclerView playLineRecyclerMessage;
    public final RecyclerView playLineRecyclerView;
    public final SmartRefreshLayout playLineRefreshLayout;
    public final TextView playLineTvLoading;
    public final AppCompatTextView playLineTxtPeoples;
    private final ConstraintLayout rootView;
    public final FrameLayout titleImgMore;
    public final LinearLayout titleLayout;
    public final TextView titleTxtName;
    public final ViewNetworkErrorBinding viewNetworkError;

    private ActivityPlayLineBinding(ConstraintLayout constraintLayout, ImageView imageView, EditText editText, ImageView imageView2, LinearLayout linearLayout, RecyclerView recyclerView, RecyclerView recyclerView2, SmartRefreshLayout smartRefreshLayout, TextView textView, AppCompatTextView appCompatTextView, FrameLayout frameLayout, LinearLayout linearLayout2, TextView textView2, ViewNetworkErrorBinding viewNetworkErrorBinding) {
        this.rootView = constraintLayout;
        this.imgPeoplesBg = imageView;
        this.playLineEditMessage = editText;
        this.playLineImgPeoples = imageView2;
        this.playLineLayoutLoading = linearLayout;
        this.playLineRecyclerMessage = recyclerView;
        this.playLineRecyclerView = recyclerView2;
        this.playLineRefreshLayout = smartRefreshLayout;
        this.playLineTvLoading = textView;
        this.playLineTxtPeoples = appCompatTextView;
        this.titleImgMore = frameLayout;
        this.titleLayout = linearLayout2;
        this.titleTxtName = textView2;
        this.viewNetworkError = viewNetworkErrorBinding;
    }

    public static ActivityPlayLineBinding bind(View view) {
        View findChildViewById;
        int i = R.id.img_peoples_bg;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.play_line_edit_message;
            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText != null) {
                i = R.id.play_line_img_peoples;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.play_line_layout_loading;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.play_line_recycler_message;
                        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                        if (recyclerView != null) {
                            i = R.id.play_line_recycler_view;
                            RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                            if (recyclerView2 != null) {
                                i = R.id.play_line_refreshLayout;
                                SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                                if (smartRefreshLayout != null) {
                                    i = R.id.play_line_tv_loading;
                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView != null) {
                                        i = R.id.play_line_txt_peoples;
                                        AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                        if (appCompatTextView != null) {
                                            i = R.id.title_img_more;
                                            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                            if (frameLayout != null) {
                                                i = R.id.title_layout;
                                                LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                if (linearLayout2 != null) {
                                                    i = R.id.title_txt_name;
                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_network_error))) != null) {
                                                        return new ActivityPlayLineBinding((ConstraintLayout) view, imageView, editText, imageView2, linearLayout, recyclerView, recyclerView2, smartRefreshLayout, textView, appCompatTextView, frameLayout, linearLayout2, textView2, ViewNetworkErrorBinding.bind(findChildViewById));
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

    public static ActivityPlayLineBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlayLineBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_line, viewGroup, false);
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
