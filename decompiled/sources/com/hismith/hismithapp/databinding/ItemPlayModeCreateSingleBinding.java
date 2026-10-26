package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemPlayModeCreateSingleBinding implements ViewBinding {
    public final ImageView itemModeCreateImgType;
    public final FrameLayout itemModeCreateLayoutDelete;
    public final LinearLayout itemModeCreateLayoutStart;
    public final LinearLayout itemModeCreateLayoutStop;
    public final LinearLayout itemModeCreateLayoutTime;
    public final TextView itemModeCreateTxtEnd;
    public final EditText itemModeCreateTxtEndValue;
    public final EditText itemModeCreateTxtEndValue2;
    public final TextView itemModeCreateTxtM1;
    public final TextView itemModeCreateTxtM2;
    public final TextView itemModeCreateTxtSerial;
    public final TextView itemModeCreateTxtStart;
    public final EditText itemModeCreateTxtStartValue;
    public final EditText itemModeCreateTxtStartValue2;
    public final TextView itemModeCreateTxtTime;
    public final TextView itemModeCreateTxtTimeValue;
    public final LinearLayout itemModeLayout;
    public final FrameLayout layoutConfirm;
    public final LinearLayout layoutSpeed;
    private final LinearLayout rootView;
    public final SeekBar seekSpeed;
    public final TextView txtSpeed;
    public final TextView txtSpeedLabel;

    private ItemPlayModeCreateSingleBinding(LinearLayout linearLayout, ImageView imageView, FrameLayout frameLayout, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, TextView textView, EditText editText, EditText editText2, TextView textView2, TextView textView3, TextView textView4, TextView textView5, EditText editText3, EditText editText4, TextView textView6, TextView textView7, LinearLayout linearLayout5, FrameLayout frameLayout2, LinearLayout linearLayout6, SeekBar seekBar, TextView textView8, TextView textView9) {
        this.rootView = linearLayout;
        this.itemModeCreateImgType = imageView;
        this.itemModeCreateLayoutDelete = frameLayout;
        this.itemModeCreateLayoutStart = linearLayout2;
        this.itemModeCreateLayoutStop = linearLayout3;
        this.itemModeCreateLayoutTime = linearLayout4;
        this.itemModeCreateTxtEnd = textView;
        this.itemModeCreateTxtEndValue = editText;
        this.itemModeCreateTxtEndValue2 = editText2;
        this.itemModeCreateTxtM1 = textView2;
        this.itemModeCreateTxtM2 = textView3;
        this.itemModeCreateTxtSerial = textView4;
        this.itemModeCreateTxtStart = textView5;
        this.itemModeCreateTxtStartValue = editText3;
        this.itemModeCreateTxtStartValue2 = editText4;
        this.itemModeCreateTxtTime = textView6;
        this.itemModeCreateTxtTimeValue = textView7;
        this.itemModeLayout = linearLayout5;
        this.layoutConfirm = frameLayout2;
        this.layoutSpeed = linearLayout6;
        this.seekSpeed = seekBar;
        this.txtSpeed = textView8;
        this.txtSpeedLabel = textView9;
    }

    public static ItemPlayModeCreateSingleBinding bind(View view) {
        int i = R.id.item_mode_create_img_type;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.item_mode_create_layout_delete;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.item_mode_create_layout_start;
                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                if (linearLayout != null) {
                    i = R.id.item_mode_create_layout_stop;
                    LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout2 != null) {
                        i = R.id.item_mode_create_layout_time;
                        LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout3 != null) {
                            i = R.id.item_mode_create_txt_end;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.item_mode_create_txt_end_value;
                                EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                                if (editText != null) {
                                    i = R.id.item_mode_create_txt_end_value_2;
                                    EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                                    if (editText2 != null) {
                                        i = R.id.item_mode_create_txt_m1;
                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView2 != null) {
                                            i = R.id.item_mode_create_txt_m2;
                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView3 != null) {
                                                i = R.id.item_mode_create_txt_serial;
                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView4 != null) {
                                                    i = R.id.item_mode_create_txt_start;
                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView5 != null) {
                                                        i = R.id.item_mode_create_txt_start_value;
                                                        EditText editText3 = (EditText) ViewBindings.findChildViewById(view, i);
                                                        if (editText3 != null) {
                                                            i = R.id.item_mode_create_txt_start_value_2;
                                                            EditText editText4 = (EditText) ViewBindings.findChildViewById(view, i);
                                                            if (editText4 != null) {
                                                                i = R.id.item_mode_create_txt_time;
                                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView6 != null) {
                                                                    i = R.id.item_mode_create_txt_time_value;
                                                                    TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView7 != null) {
                                                                        i = R.id.item_mode_layout;
                                                                        LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (linearLayout4 != null) {
                                                                            i = R.id.layout_confirm;
                                                                            FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                            if (frameLayout2 != null) {
                                                                                i = R.id.layout_speed;
                                                                                LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                if (linearLayout5 != null) {
                                                                                    i = R.id.seek_speed;
                                                                                    SeekBar seekBar = (SeekBar) ViewBindings.findChildViewById(view, i);
                                                                                    if (seekBar != null) {
                                                                                        i = R.id.txt_speed;
                                                                                        TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView8 != null) {
                                                                                            i = R.id.txt_speed_label;
                                                                                            TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView9 != null) {
                                                                                                return new ItemPlayModeCreateSingleBinding((LinearLayout) view, imageView, frameLayout, linearLayout, linearLayout2, linearLayout3, textView, editText, editText2, textView2, textView3, textView4, textView5, editText3, editText4, textView6, textView7, linearLayout4, frameLayout2, linearLayout5, seekBar, textView8, textView9);
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

    public static ItemPlayModeCreateSingleBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPlayModeCreateSingleBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_play_mode_create_single, viewGroup, false);
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
