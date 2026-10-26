package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityLanguageBinding implements ViewBinding {
    public final TitleBarBinding include;
    public final RecyclerView languageRecyclerView;
    private final LinearLayout rootView;

    private ActivityLanguageBinding(LinearLayout linearLayout, TitleBarBinding titleBarBinding, RecyclerView recyclerView) {
        this.rootView = linearLayout;
        this.include = titleBarBinding;
        this.languageRecyclerView = recyclerView;
    }

    public static ActivityLanguageBinding bind(View view) {
        int i = R.id.include;
        View findChildViewById = ViewBindings.findChildViewById(view, i);
        if (findChildViewById != null) {
            TitleBarBinding bind = TitleBarBinding.bind(findChildViewById);
            int i2 = R.id.language_recyclerView;
            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i2);
            if (recyclerView != null) {
                return new ActivityLanguageBinding((LinearLayout) view, bind, recyclerView);
            }
            i = i2;
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityLanguageBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityLanguageBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_language, viewGroup, false);
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
