package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityBlackListBinding implements ViewBinding {
    public final SlideRecyclerView blackRecycler;
    public final SmartRefreshLayout blackRefreshLayout;
    public final TextView blackTxtSend;
    public final TitleBarBinding include;
    private final LinearLayout rootView;

    private ActivityBlackListBinding(LinearLayout linearLayout, SlideRecyclerView slideRecyclerView, SmartRefreshLayout smartRefreshLayout, TextView textView, TitleBarBinding titleBarBinding) {
        this.rootView = linearLayout;
        this.blackRecycler = slideRecyclerView;
        this.blackRefreshLayout = smartRefreshLayout;
        this.blackTxtSend = textView;
        this.include = titleBarBinding;
    }

    public static ActivityBlackListBinding bind(View view) {
        View findChildViewById;
        int i = R.id.black_recycler;
        SlideRecyclerView slideRecyclerView = (SlideRecyclerView) ViewBindings.findChildViewById(view, i);
        if (slideRecyclerView != null) {
            i = R.id.black_refreshLayout;
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
            if (smartRefreshLayout != null) {
                i = R.id.black_txt_send;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.include))) != null) {
                    return new ActivityBlackListBinding((LinearLayout) view, slideRecyclerView, smartRefreshLayout, textView, TitleBarBinding.bind(findChildViewById));
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityBlackListBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityBlackListBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_black_list, viewGroup, false);
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
