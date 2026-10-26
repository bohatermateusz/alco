package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.LinePointsView;

/* loaded from: classes3.dex */
public final class ItemPlayModeBinding implements ViewBinding {
    public final ImageView imgHasPiu;
    public final ImageView itemModeImgCollectCount;
    public final CircleImageView itemModeImgDevice;
    public final CircleImageView itemModeImgHead;
    public final ImageView itemModeImgLikeCount;
    public final ImageView itemModeImgMore;
    public final ImageView itemModeImgPlayCount;
    public final ImageView itemModeImgTime;
    public final LinearLayout itemModeLayoutAction;
    public final LinePointsView itemModeLinePoints;
    public final TextView itemModeTxtCollectCount;
    public final TextView itemModeTxtLikeCount;
    public final TextView itemModeTxtName;
    public final TextView itemModeTxtPlayCount;
    public final TextView itemModeTxtTime;
    public final LinearLayout layoutSetting;
    private final FrameLayout rootView;
    public final TextView txtSettingPositionMax;
    public final TextView txtSettingPositionMin;
    public final TextView txtSettingSmooth;

    private ItemPlayModeBinding(FrameLayout frameLayout, ImageView imageView, ImageView imageView2, CircleImageView circleImageView, CircleImageView circleImageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, LinearLayout linearLayout, LinePointsView linePointsView, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, LinearLayout linearLayout2, TextView textView6, TextView textView7, TextView textView8) {
        this.rootView = frameLayout;
        this.imgHasPiu = imageView;
        this.itemModeImgCollectCount = imageView2;
        this.itemModeImgDevice = circleImageView;
        this.itemModeImgHead = circleImageView2;
        this.itemModeImgLikeCount = imageView3;
        this.itemModeImgMore = imageView4;
        this.itemModeImgPlayCount = imageView5;
        this.itemModeImgTime = imageView6;
        this.itemModeLayoutAction = linearLayout;
        this.itemModeLinePoints = linePointsView;
        this.itemModeTxtCollectCount = textView;
        this.itemModeTxtLikeCount = textView2;
        this.itemModeTxtName = textView3;
        this.itemModeTxtPlayCount = textView4;
        this.itemModeTxtTime = textView5;
        this.layoutSetting = linearLayout2;
        this.txtSettingPositionMax = textView6;
        this.txtSettingPositionMin = textView7;
        this.txtSettingSmooth = textView8;
    }

    public static ItemPlayModeBinding bind(View view) {
        int i = R.id.img_has_piu;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_mode_img_collectCount;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.item_mode_img_device;
                CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                if (circleImageView != null) {
                    i = R.id.item_mode_img_head;
                    CircleImageView circleImageView2 = (CircleImageView) ViewBindings.findChildViewById(view, i);
                    if (circleImageView2 != null) {
                        i = R.id.item_mode_img_likeCount;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.item_mode_img_more;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.item_mode_img_playCount;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.item_mode_img_time;
                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView6 != null) {
                                        i = R.id.item_mode_layout_action;
                                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout != null) {
                                            i = R.id.item_mode_LinePoints;
                                            LinePointsView linePointsView = (LinePointsView) ViewBindings.findChildViewById(view, i);
                                            if (linePointsView != null) {
                                                i = R.id.item_mode_txt_collectCount;
                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView != null) {
                                                    i = R.id.item_mode_txt_likeCount;
                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView2 != null) {
                                                        i = R.id.item_mode_txt_name;
                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView3 != null) {
                                                            i = R.id.item_mode_txt_playCount;
                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView4 != null) {
                                                                i = R.id.item_mode_txt_time;
                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView5 != null) {
                                                                    i = R.id.layout_setting;
                                                                    LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (linearLayout2 != null) {
                                                                        i = R.id.txt_setting_position_max;
                                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView6 != null) {
                                                                            i = R.id.txt_setting_position_min;
                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView7 != null) {
                                                                                i = R.id.txt_setting_smooth;
                                                                                TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView8 != null) {
                                                                                    return new ItemPlayModeBinding((FrameLayout) view, imageView, imageView2, circleImageView, circleImageView2, imageView3, imageView4, imageView5, imageView6, linearLayout, linePointsView, textView, textView2, textView3, textView4, textView5, linearLayout2, textView6, textView7, textView8);
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

    public static ItemPlayModeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlayModeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_mode, viewGroup, false);
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
