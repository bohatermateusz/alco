package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityAddFriendBinding implements ViewBinding {
    public final EditText addEditSearch;
    public final ImageView addImgSearch;
    public final LinearLayout addLayoutSearch;
    public final RecyclerView addRecycler;
    public final SmartRefreshLayout addRefreshLayout;
    public final TextView addTxtCancel;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityAddFriendBinding(LinearLayout linearLayout, EditText editText, ImageView imageView, LinearLayout linearLayout2, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.addEditSearch = editText;
        this.addImgSearch = imageView;
        this.addLayoutSearch = linearLayout2;
        this.addRecycler = recyclerView;
        this.addRefreshLayout = smartRefreshLayout;
        this.addTxtCancel = textView;
        this.titleTxtName = textView2;
    }

    public static ActivityAddFriendBinding bind(View view) {
        int i = R.id.add_edit_search;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.add_img_search;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.add_layout_search;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.add_recycler;
                    RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                    if (recyclerView != null) {
                        i = R.id.add_refreshLayout;
                        SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                        if (smartRefreshLayout != null) {
                            i = R.id.add_txt_cancel;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.title_txt_name;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    return new ActivityAddFriendBinding((LinearLayout) view, editText, imageView, linearLayout, recyclerView, smartRefreshLayout, textView, textView2);
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityAddFriendBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityAddFriendBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_add_friend, viewGroup, false);
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
