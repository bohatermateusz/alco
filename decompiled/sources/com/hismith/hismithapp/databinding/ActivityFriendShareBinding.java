package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityFriendShareBinding implements ViewBinding {
    public final EditText friendShareEditSearch;
    public final FrameLayout friendShareLayoutSelect;
    public final RecyclerView friendShareRecyclerView;
    public final SmartRefreshLayout friendShareRefreshLayout;
    public final TextView friendShareTxtContent;
    public final TextView friendShareTxtContentLabel;
    public final TextView friendShareTxtRecent;
    public final TextView friendShareTxtSend;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityFriendShareBinding(LinearLayout linearLayout, EditText editText, FrameLayout frameLayout, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5) {
        this.rootView = linearLayout;
        this.friendShareEditSearch = editText;
        this.friendShareLayoutSelect = frameLayout;
        this.friendShareRecyclerView = recyclerView;
        this.friendShareRefreshLayout = smartRefreshLayout;
        this.friendShareTxtContent = textView;
        this.friendShareTxtContentLabel = textView2;
        this.friendShareTxtRecent = textView3;
        this.friendShareTxtSend = textView4;
        this.titleTxtName = textView5;
    }

    public static ActivityFriendShareBinding bind(View view) {
        int i = R.id.friend_share_edit_search;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.friend_share_layout_select;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.friend_share_recycler_view;
                RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                if (recyclerView != null) {
                    i = R.id.friend_share_refreshLayout;
                    SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                    if (smartRefreshLayout != null) {
                        i = R.id.friend_share_txt_content;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.friend_share_txt_content_label;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.friend_share_txt_recent;
                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView3 != null) {
                                    i = R.id.friend_share_txt_send;
                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView4 != null) {
                                        i = R.id.title_txt_name;
                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView5 != null) {
                                            return new ActivityFriendShareBinding((LinearLayout) view, editText, frameLayout, recyclerView, smartRefreshLayout, textView, textView2, textView3, textView4, textView5);
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

    public static ActivityFriendShareBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityFriendShareBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_friend_share, viewGroup, false);
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
