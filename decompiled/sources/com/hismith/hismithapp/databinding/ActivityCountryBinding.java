package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityCountryBinding implements ViewBinding {
    public final RecyclerView countryRecycler;
    public final TitleBarBinding include;
    private final LinearLayout rootView;
    public final TextView txtAll;
    public final TextView txtCountry;
    public final TextView txtTarget;

    private ActivityCountryBinding(LinearLayout linearLayout, RecyclerView recyclerView, TitleBarBinding titleBarBinding, TextView textView, TextView textView2, TextView textView3) {
        this.rootView = linearLayout;
        this.countryRecycler = recyclerView;
        this.include = titleBarBinding;
        this.txtAll = textView;
        this.txtCountry = textView2;
        this.txtTarget = textView3;
    }

    public static ActivityCountryBinding bind(View view) {
        View findChildViewById;
        int i = R.id.country_recycler;
        RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
        if (recyclerView != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.include))) != null) {
            TitleBarBinding bind = TitleBarBinding.bind(findChildViewById);
            i = R.id.txt_all;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                i = R.id.txt_country;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.txt_target;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        return new ActivityCountryBinding((LinearLayout) view, recyclerView, bind, textView, textView2, textView3);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityCountryBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityCountryBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_country, viewGroup, false);
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
