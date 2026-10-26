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
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityChatBinding implements ViewBinding {
    public final EditText chatEditMessage;
    public final ImageView chatImgEmoji;
    public final ImageView chatImgFun;
    public final ImageView chatImgVoice;
    public final ImageView chatImgVoiceStatus;
    public final ConstraintLayout chatLayoutAction;
    public final FrameLayout chatLayoutEdit;
    public final LinearLayout chatLayoutEmoji;
    public final LinearLayout chatLayoutFun;
    public final LinearLayout chatLayoutVoiceStatus;
    public final RecyclerView chatRecyclerEmoji;
    public final RecyclerView chatRecyclerFun;
    public final RecyclerView chatRecyclerView;
    public final ImageView chatReferenceClose;
    public final TextView chatReferenceContent;
    public final ConstraintLayout chatReferenceLayout;
    public final SmartRefreshLayout chatRefreshLayout;
    public final TextView chatTxtVoice;
    public final TextView chatTxtVoiceStatus;
    private final FrameLayout rootView;
    public final ImageView titleImgMore;
    public final TextView titleTxtName;

    private ActivityChatBinding(FrameLayout frameLayout, EditText editText, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ConstraintLayout constraintLayout, FrameLayout frameLayout2, LinearLayout linearLayout, LinearLayout linearLayout2, LinearLayout linearLayout3, RecyclerView recyclerView, RecyclerView recyclerView2, RecyclerView recyclerView3, ImageView imageView5, TextView textView, ConstraintLayout constraintLayout2, SmartRefreshLayout smartRefreshLayout, TextView textView2, TextView textView3, ImageView imageView6, TextView textView4) {
        this.rootView = frameLayout;
        this.chatEditMessage = editText;
        this.chatImgEmoji = imageView;
        this.chatImgFun = imageView2;
        this.chatImgVoice = imageView3;
        this.chatImgVoiceStatus = imageView4;
        this.chatLayoutAction = constraintLayout;
        this.chatLayoutEdit = frameLayout2;
        this.chatLayoutEmoji = linearLayout;
        this.chatLayoutFun = linearLayout2;
        this.chatLayoutVoiceStatus = linearLayout3;
        this.chatRecyclerEmoji = recyclerView;
        this.chatRecyclerFun = recyclerView2;
        this.chatRecyclerView = recyclerView3;
        this.chatReferenceClose = imageView5;
        this.chatReferenceContent = textView;
        this.chatReferenceLayout = constraintLayout2;
        this.chatRefreshLayout = smartRefreshLayout;
        this.chatTxtVoice = textView2;
        this.chatTxtVoiceStatus = textView3;
        this.titleImgMore = imageView6;
        this.titleTxtName = textView4;
    }

    public static ActivityChatBinding bind(View view) {
        int i = R.id.chat_edit_message;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.chat_img_emoji;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.chat_img_fun;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.chat_img_voice;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.chat_img_voice_status;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.chat_layout_action;
                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout != null) {
                                i = R.id.chat_layout_edit;
                                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                if (frameLayout != null) {
                                    i = R.id.chat_layout_emoji;
                                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout != null) {
                                        i = R.id.chat_layout_fun;
                                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout2 != null) {
                                            i = R.id.chat_layout_voice_status;
                                            LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout3 != null) {
                                                i = R.id.chat_recycler_emoji;
                                                RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                if (recyclerView != null) {
                                                    i = R.id.chat_recycler_fun;
                                                    RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                    if (recyclerView2 != null) {
                                                        i = R.id.chat_recyclerView;
                                                        RecyclerView recyclerView3 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                        if (recyclerView3 != null) {
                                                            i = R.id.chat_reference_close;
                                                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView5 != null) {
                                                                i = R.id.chat_reference_content;
                                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView != null) {
                                                                    i = R.id.chat_reference_layout;
                                                                    ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (constraintLayout2 != null) {
                                                                        i = R.id.chat_refreshLayout;
                                                                        SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (smartRefreshLayout != null) {
                                                                            i = R.id.chat_txt_voice;
                                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView2 != null) {
                                                                                i = R.id.chat_txt_voice_status;
                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView3 != null) {
                                                                                    i = R.id.title_img_more;
                                                                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                    if (imageView6 != null) {
                                                                                        i = R.id.title_txt_name;
                                                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView4 != null) {
                                                                                            return new ActivityChatBinding((FrameLayout) view, editText, imageView, imageView2, imageView3, imageView4, constraintLayout, frameLayout, linearLayout, linearLayout2, linearLayout3, recyclerView, recyclerView2, recyclerView3, imageView5, textView, constraintLayout2, smartRefreshLayout, textView2, textView3, imageView6, textView4);
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

    public static ActivityChatBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityChatBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_chat, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public FrameLayout getRoot() {
        return this.rootView;
    }
}
