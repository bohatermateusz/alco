package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ItemChatLeftBinding implements ViewBinding {
    public final ImageView chatItemImgContent;
    public final ImageView chatItemImgError;
    public final CircleImageView chatItemImgHead;
    public final ImageView chatItemImgPlay;
    public final FrameLayout chatItemLayout;
    public final FrameLayout chatItemLayoutStatus;
    public final FrameLayout chatItemLayoutVoice;
    public final ProgressBar chatItemProgress;
    public final TextView chatItemTxtContent;
    public final TextView chatItemTxtSendTime;
    public final TextView chatItemTxtTime;
    public final ItemChatReferenceBinding chatLayoutReference;
    private final ConstraintLayout rootView;

    private ItemChatLeftBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, ImageView imageView3, FrameLayout frameLayout, FrameLayout frameLayout2, FrameLayout frameLayout3, ProgressBar progressBar, TextView textView, TextView textView2, TextView textView3, ItemChatReferenceBinding itemChatReferenceBinding) {
        this.rootView = constraintLayout;
        this.chatItemImgContent = imageView;
        this.chatItemImgError = imageView2;
        this.chatItemImgHead = circleImageView;
        this.chatItemImgPlay = imageView3;
        this.chatItemLayout = frameLayout;
        this.chatItemLayoutStatus = frameLayout2;
        this.chatItemLayoutVoice = frameLayout3;
        this.chatItemProgress = progressBar;
        this.chatItemTxtContent = textView;
        this.chatItemTxtSendTime = textView2;
        this.chatItemTxtTime = textView3;
        this.chatLayoutReference = itemChatReferenceBinding;
    }

    public static ItemChatLeftBinding bind(View view) {
        View findChildViewById;
        int i = R.id.chat_item_img_content;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.chat_item_img_error;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.chat_item_img_head;
                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                if (circleImageView != null) {
                    i = R.id.chat_item_img_play;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.chat_item_layout;
                        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                        if (frameLayout != null) {
                            i = R.id.chat_item_layout_status;
                            FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                            if (frameLayout2 != null) {
                                i = R.id.chat_item_layout_voice;
                                FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                if (frameLayout3 != null) {
                                    i = R.id.chat_item_progress;
                                    ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                    if (progressBar != null) {
                                        i = R.id.chat_item_txt_content;
                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView != null) {
                                            i = R.id.chat_item_txt_send_time;
                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView2 != null) {
                                                i = R.id.chat_item_txt_time;
                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView3 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.chat_layout_reference))) != null) {
                                                    return new ItemChatLeftBinding((ConstraintLayout) view, imageView, imageView2, circleImageView, imageView3, frameLayout, frameLayout2, frameLayout3, progressBar, textView, textView2, textView3, ItemChatReferenceBinding.bind(findChildViewById));
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

    public static ItemChatLeftBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemChatLeftBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_chat_left, viewGroup, false);
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
