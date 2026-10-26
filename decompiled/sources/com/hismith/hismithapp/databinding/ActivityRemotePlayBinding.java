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
import com.hismith.hismithapp.view.ClipLayout;

/* loaded from: classes3.dex */
public final class ActivityRemotePlayBinding implements ViewBinding {
    public final FrameLayout frameLayout;
    public final ImageView imgSetting;
    public final ClipLayout remoteChatClipLayout;
    public final EditText remoteEditMessage;
    public final ImageView remoteImgChat;
    public final ConstraintLayout remoteLayoutChat;
    public final LinearLayout remoteLayoutMessage;
    public final ImageView remotePlayImgBg;
    public final ImageView remotePlayImgCloseLeft;
    public final ImageView remotePlayImgCloseRight;
    public final ImageView remotePlayImgFloat;
    public final ImageView remotePlayImgPiu;
    public final LinearLayout remotePlayLayoutTitle;
    public final RecyclerView remoteRecyclerChat;
    private final ConstraintLayout rootView;
    public final TextView titleTxtName;
    public final TextView titleTxtStatus;
    public final ViewNetworkErrorBinding viewNetworkError;

    private ActivityRemotePlayBinding(ConstraintLayout constraintLayout, FrameLayout frameLayout, ImageView imageView, ClipLayout clipLayout, EditText editText, ImageView imageView2, ConstraintLayout constraintLayout2, LinearLayout linearLayout, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, LinearLayout linearLayout2, RecyclerView recyclerView, TextView textView, TextView textView2, ViewNetworkErrorBinding viewNetworkErrorBinding) {
        this.rootView = constraintLayout;
        this.frameLayout = frameLayout;
        this.imgSetting = imageView;
        this.remoteChatClipLayout = clipLayout;
        this.remoteEditMessage = editText;
        this.remoteImgChat = imageView2;
        this.remoteLayoutChat = constraintLayout2;
        this.remoteLayoutMessage = linearLayout;
        this.remotePlayImgBg = imageView3;
        this.remotePlayImgCloseLeft = imageView4;
        this.remotePlayImgCloseRight = imageView5;
        this.remotePlayImgFloat = imageView6;
        this.remotePlayImgPiu = imageView7;
        this.remotePlayLayoutTitle = linearLayout2;
        this.remoteRecyclerChat = recyclerView;
        this.titleTxtName = textView;
        this.titleTxtStatus = textView2;
        this.viewNetworkError = viewNetworkErrorBinding;
    }

    public static ActivityRemotePlayBinding bind(View view) {
        View findChildViewById;
        int i = R.id.frame_layout;
        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
        if (frameLayout != null) {
            i = R.id.img_setting;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.remote_chat_clip_layout;
                ClipLayout clipLayout = (ClipLayout) ViewBindings.findChildViewById(view, i);
                if (clipLayout != null) {
                    i = R.id.remote_edit_message;
                    EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText != null) {
                        i = R.id.remote_img_chat;
                        ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView2 != null) {
                            i = R.id.remote_layout_chat;
                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout != null) {
                                i = R.id.remote_layout_message;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.remote_play_img_bg;
                                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView3 != null) {
                                        i = R.id.remote_play_img_close_left;
                                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView4 != null) {
                                            i = R.id.remote_play_img_close_right;
                                            ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView5 != null) {
                                                i = R.id.remote_play_img_float;
                                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView6 != null) {
                                                    i = R.id.remote_play_img_piu;
                                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                    if (imageView7 != null) {
                                                        i = R.id.remote_play_layout_title;
                                                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                        if (linearLayout2 != null) {
                                                            i = R.id.remote_recycler_chat;
                                                            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                            if (recyclerView != null) {
                                                                i = R.id.title_txt_name;
                                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView != null) {
                                                                    i = R.id.title_txt_status;
                                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_network_error))) != null) {
                                                                        return new ActivityRemotePlayBinding((ConstraintLayout) view, frameLayout, imageView, clipLayout, editText, imageView2, constraintLayout, linearLayout, imageView3, imageView4, imageView5, imageView6, imageView7, linearLayout2, recyclerView, textView, textView2, ViewNetworkErrorBinding.bind(findChildViewById));
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

    public static ActivityRemotePlayBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityRemotePlayBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_remote_play, viewGroup, false);
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
