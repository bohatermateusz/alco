package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.widget.NestedScrollView;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityPlayModeCreateBinding implements ViewBinding {
    public final ImageView imgLoop;
    public final ImageView imgReversal;
    public final ImageView imgSingle;
    public final LinearLayout layoutLoop;
    public final LinearLayout layoutPlug;
    public final LinearLayout layoutReversal;
    public final LinearLayout layoutSingle;
    public final ImageView modeCreateImgCreate;
    public final ImageView modeCreateImgPreview;
    public final LinearLayout modeCreateLayout;
    public final LinearLayout modeCreateLayoutComplete;
    public final LinearLayout modeCreateLayoutCreate;
    public final LinearLayout modeCreateLayoutPopup;
    public final LinearLayout modeCreateLayoutPreview;
    public final RecyclerView modeCreateRecyclerView;
    public final TextView modeCreateTxtComplete;
    public final TextView modeCreateTxtCreate;
    public final TextView modeCreateTxtPreview;
    private final LinearLayout rootView;
    public final NestedScrollView scrollView;
    public final ImageView titleImgBluetooth;
    public final LinearLayout titleLayout;
    public final TextView titleTxtName;
    public final TextView txtLoop;
    public final TextView txtReversal;
    public final TextView txtSingle;

    private ActivityPlayModeCreateBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, LinearLayout linearLayout5, ImageView imageView4, ImageView imageView5, LinearLayout linearLayout6, LinearLayout linearLayout7, LinearLayout linearLayout8, LinearLayout linearLayout9, LinearLayout linearLayout10, RecyclerView recyclerView, TextView textView, TextView textView2, TextView textView3, NestedScrollView nestedScrollView, ImageView imageView6, LinearLayout linearLayout11, TextView textView4, TextView textView5, TextView textView6, TextView textView7) {
        this.rootView = linearLayout;
        this.imgLoop = imageView;
        this.imgReversal = imageView2;
        this.imgSingle = imageView3;
        this.layoutLoop = linearLayout2;
        this.layoutPlug = linearLayout3;
        this.layoutReversal = linearLayout4;
        this.layoutSingle = linearLayout5;
        this.modeCreateImgCreate = imageView4;
        this.modeCreateImgPreview = imageView5;
        this.modeCreateLayout = linearLayout6;
        this.modeCreateLayoutComplete = linearLayout7;
        this.modeCreateLayoutCreate = linearLayout8;
        this.modeCreateLayoutPopup = linearLayout9;
        this.modeCreateLayoutPreview = linearLayout10;
        this.modeCreateRecyclerView = recyclerView;
        this.modeCreateTxtComplete = textView;
        this.modeCreateTxtCreate = textView2;
        this.modeCreateTxtPreview = textView3;
        this.scrollView = nestedScrollView;
        this.titleImgBluetooth = imageView6;
        this.titleLayout = linearLayout11;
        this.titleTxtName = textView4;
        this.txtLoop = textView5;
        this.txtReversal = textView6;
        this.txtSingle = textView7;
    }

    public static ActivityPlayModeCreateBinding bind(View view) {
        int i = R.id.img_loop;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_reversal;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_single;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.layout_loop;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.layout_plug;
                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout2 != null) {
                            i = R.id.layout_reversal;
                            LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout3 != null) {
                                i = R.id.layout_single;
                                LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout4 != null) {
                                    i = R.id.mode_create_img_create;
                                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView4 != null) {
                                        i = R.id.mode_create_img_preview;
                                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView5 != null) {
                                            i = R.id.mode_create_layout;
                                            LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout5 != null) {
                                                i = R.id.mode_create_layout_complete;
                                                LinearLayout linearLayout6 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                if (linearLayout6 != null) {
                                                    i = R.id.mode_create_layout_create;
                                                    LinearLayout linearLayout7 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                    if (linearLayout7 != null) {
                                                        i = R.id.mode_create_layout_popup;
                                                        LinearLayout linearLayout8 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                        if (linearLayout8 != null) {
                                                            i = R.id.mode_create_layout_preview;
                                                            LinearLayout linearLayout9 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                            if (linearLayout9 != null) {
                                                                i = R.id.mode_create_recycler_view;
                                                                RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                if (recyclerView != null) {
                                                                    i = R.id.mode_create_txt_complete;
                                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView != null) {
                                                                        i = R.id.mode_create_txt_create;
                                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView2 != null) {
                                                                            i = R.id.mode_create_txt_preview;
                                                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView3 != null) {
                                                                                i = R.id.scroll_view;
                                                                                NestedScrollView nestedScrollView = (NestedScrollView) ViewBindings.findChildViewById(view, i);
                                                                                if (nestedScrollView != null) {
                                                                                    i = R.id.title_img_bluetooth;
                                                                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                    if (imageView6 != null) {
                                                                                        i = R.id.title_layout;
                                                                                        LinearLayout linearLayout10 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                        if (linearLayout10 != null) {
                                                                                            i = R.id.title_txt_name;
                                                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView4 != null) {
                                                                                                i = R.id.txt_loop;
                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView5 != null) {
                                                                                                    i = R.id.txt_reversal;
                                                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView6 != null) {
                                                                                                        i = R.id.txt_single;
                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView7 != null) {
                                                                                                            return new ActivityPlayModeCreateBinding((LinearLayout) view, imageView, imageView2, imageView3, linearLayout, linearLayout2, linearLayout3, linearLayout4, imageView4, imageView5, linearLayout5, linearLayout6, linearLayout7, linearLayout8, linearLayout9, recyclerView, textView, textView2, textView3, nestedScrollView, imageView6, linearLayout10, textView4, textView5, textView6, textView7);
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
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPlayModeCreateBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlayModeCreateBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_mode_create, viewGroup, false);
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
