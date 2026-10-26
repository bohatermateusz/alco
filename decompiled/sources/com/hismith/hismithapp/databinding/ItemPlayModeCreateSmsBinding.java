package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemPlayModeCreateSmsBinding implements ViewBinding {
    public final ImageView itemModeCreateImgType;
    public final FrameLayout itemModeCreateLayoutDelete;
    public final LinearLayout itemModeCreateLayoutEnd;
    public final LinearLayout itemModeCreateLayoutEndSpeed;
    public final LinearLayout itemModeCreateLayoutSmooth;
    public final LinearLayout itemModeCreateLayoutStart;
    public final LinearLayout itemModeCreateLayoutStartSpeed;
    public final LinearLayout itemModeCreateLayoutTime;
    public final TextView itemModeCreateTxtEnd;
    public final EditText itemModeCreateTxtEndPosition;
    public final TextView itemModeCreateTxtEndSpeed;
    public final EditText itemModeCreateTxtEndSpeedValue;
    public final TextView itemModeCreateTxtSerial;
    public final TextView itemModeCreateTxtSmooth;
    public final TextView itemModeCreateTxtSmoothValue;
    public final TextView itemModeCreateTxtStart;
    public final EditText itemModeCreateTxtStartPosition;
    public final TextView itemModeCreateTxtStartSpeed;
    public final EditText itemModeCreateTxtStartSpeedValue;
    public final TextView itemModeCreateTxtTime;
    public final TextView itemModeCreateTxtTimeValue;
    public final LinearLayout itemModeLayout;
    private final LinearLayout rootView;

    private ItemPlayModeCreateSmsBinding(LinearLayout linearLayout, ImageView imageView, FrameLayout frameLayout, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, LinearLayout linearLayout5, LinearLayout linearLayout6, LinearLayout linearLayout7, TextView textView, EditText editText, TextView textView2, EditText editText2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, EditText editText3, TextView textView7, EditText editText4, TextView textView8, TextView textView9, LinearLayout linearLayout8) {
        this.rootView = linearLayout;
        this.itemModeCreateImgType = imageView;
        this.itemModeCreateLayoutDelete = frameLayout;
        this.itemModeCreateLayoutEnd = linearLayout2;
        this.itemModeCreateLayoutEndSpeed = linearLayout3;
        this.itemModeCreateLayoutSmooth = linearLayout4;
        this.itemModeCreateLayoutStart = linearLayout5;
        this.itemModeCreateLayoutStartSpeed = linearLayout6;
        this.itemModeCreateLayoutTime = linearLayout7;
        this.itemModeCreateTxtEnd = textView;
        this.itemModeCreateTxtEndPosition = editText;
        this.itemModeCreateTxtEndSpeed = textView2;
        this.itemModeCreateTxtEndSpeedValue = editText2;
        this.itemModeCreateTxtSerial = textView3;
        this.itemModeCreateTxtSmooth = textView4;
        this.itemModeCreateTxtSmoothValue = textView5;
        this.itemModeCreateTxtStart = textView6;
        this.itemModeCreateTxtStartPosition = editText3;
        this.itemModeCreateTxtStartSpeed = textView7;
        this.itemModeCreateTxtStartSpeedValue = editText4;
        this.itemModeCreateTxtTime = textView8;
        this.itemModeCreateTxtTimeValue = textView9;
        this.itemModeLayout = linearLayout8;
    }

    public static ItemPlayModeCreateSmsBinding bind(View view) {
        int i = R.id.item_mode_create_img_type;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_mode_create_layout_delete;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.item_mode_create_layout_end;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.item_mode_create_layout_end_speed;
                    LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout2 != null) {
                        i = R.id.item_mode_create_layout_smooth;
                        LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout3 != null) {
                            i = R.id.item_mode_create_layout_start;
                            LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout4 != null) {
                                i = R.id.item_mode_create_layout_start_speed;
                                LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout5 != null) {
                                    i = R.id.item_mode_create_layout_time;
                                    LinearLayout linearLayout6 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout6 != null) {
                                        i = R.id.item_mode_create_txt_end;
                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView != null) {
                                            i = R.id.item_mode_create_txt_end_position;
                                            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                                            if (editText != null) {
                                                i = R.id.item_mode_create_txt_end_speed;
                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView2 != null) {
                                                    i = R.id.item_mode_create_txt_end_speed_value;
                                                    EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                                                    if (editText2 != null) {
                                                        i = R.id.item_mode_create_txt_serial;
                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView3 != null) {
                                                            i = R.id.item_mode_create_txt_smooth;
                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView4 != null) {
                                                                i = R.id.item_mode_create_txt_smooth_value;
                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView5 != null) {
                                                                    i = R.id.item_mode_create_txt_start;
                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView6 != null) {
                                                                        i = R.id.item_mode_create_txt_start_position;
                                                                        EditText editText3 = (EditText) ViewBindings.findChildViewById(view, i);
                                                                        if (editText3 != null) {
                                                                            i = R.id.item_mode_create_txt_start_speed;
                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView7 != null) {
                                                                                i = R.id.item_mode_create_txt_start_speed_value;
                                                                                EditText editText4 = (EditText) ViewBindings.findChildViewById(view, i);
                                                                                if (editText4 != null) {
                                                                                    i = R.id.item_mode_create_txt_time;
                                                                                    TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView8 != null) {
                                                                                        i = R.id.item_mode_create_txt_time_value;
                                                                                        TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView9 != null) {
                                                                                            i = R.id.item_mode_layout;
                                                                                            LinearLayout linearLayout7 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                            if (linearLayout7 != null) {
                                                                                                return new ItemPlayModeCreateSmsBinding((LinearLayout) view, imageView, frameLayout, linearLayout, linearLayout2, linearLayout3, linearLayout4, linearLayout5, linearLayout6, textView, editText, textView2, editText2, textView3, textView4, textView5, textView6, editText3, textView7, editText4, textView8, textView9, linearLayout7);
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

    public static ItemPlayModeCreateSmsBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlayModeCreateSmsBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_mode_create_sms, viewGroup, false);
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
