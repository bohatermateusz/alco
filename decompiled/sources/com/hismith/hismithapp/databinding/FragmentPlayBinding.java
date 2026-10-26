package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class FragmentPlayBinding implements ViewBinding {
    public final ImageView bgBattery;
    public final ImageView imgBattery;
    public final ConstraintLayout layoutBattery;
    public final ConstraintLayout linearLayout;
    public final View playImgBg;
    public final ImageView playImgBluetooth;
    public final ImageView playImgDevice;
    public final ImageView playImgPlay;
    public final ConstraintLayout playLayout;
    public final RecyclerView playRecyclerView;
    public final SmartRefreshLayout playRefreshLayout;
    public final TextView playTxtBluetooth;
    private final ConstraintLayout rootView;
    public final View viewBattery;

    private FragmentPlayBinding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, View view, ImageView imageView3, ImageView imageView4, ImageView imageView5, ConstraintLayout constraintLayout4, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, TextView textView, View view2) {
        this.rootView = constraintLayout;
        this.bgBattery = imageView;
        this.imgBattery = imageView2;
        this.layoutBattery = constraintLayout2;
        this.linearLayout = constraintLayout3;
        this.playImgBg = view;
        this.playImgBluetooth = imageView3;
        this.playImgDevice = imageView4;
        this.playImgPlay = imageView5;
        this.playLayout = constraintLayout4;
        this.playRecyclerView = recyclerView;
        this.playRefreshLayout = smartRefreshLayout;
        this.playTxtBluetooth = textView;
        this.viewBattery = view2;
    }

    public static FragmentPlayBinding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        int i = R.id.bg_battery;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_battery;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.layout_battery;
                ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                if (constraintLayout != null) {
                    i = R.id.linearLayout;
                    ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                    if (constraintLayout2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.play_img_bg))) != null) {
                        i = R.id.play_img_bluetooth;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.play_img_device;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.play_img_play;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    ConstraintLayout constraintLayout3 = (ConstraintLayout) view;
                                    i = R.id.play_recycler_view;
                                    RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                    if (recyclerView != null) {
                                        i = R.id.play_refreshLayout;
                                        SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                                        if (smartRefreshLayout != null) {
                                            i = R.id.play_txt_bluetooth;
                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.view_battery))) != null) {
                                                return new FragmentPlayBinding(constraintLayout3, imageView, imageView2, constraintLayout, constraintLayout2, findChildViewById, imageView3, imageView4, imageView5, constraintLayout3, recyclerView, smartRefreshLayout, textView, findChildViewById2);
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

    public static FragmentPlayBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentPlayBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_play, viewGroup, false);
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
