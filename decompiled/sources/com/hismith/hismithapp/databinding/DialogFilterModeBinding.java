package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.google.android.flexbox.FlexboxLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.FitHeightTextView;

/* loaded from: classes3.dex */
public final class DialogFilterModeBinding implements ViewBinding {
    public final EditText editEndTime;
    public final EditText editName;
    public final EditText editPlayCountEnd;
    public final EditText editPlayCountStart;
    public final EditText editStartTime;
    public final ImageView imgMaxSpeed;
    public final ImageView imgSortPlayCount;
    public final ImageView imgSortTime;
    public final LinearLayout layoutFilterOnly;
    public final ConstraintLayout layoutMaxSpeed;
    public final FlexboxLayout layoutReleaseTime;
    public final FlexboxLayout layoutSortBy;
    public final ConstraintLayout layoutSortPlayCount;
    public final ConstraintLayout layoutSortTime;
    public final Guideline lineMaxStart;
    public final ProgressBar progressSpeed;
    private final ConstraintLayout rootView;
    public final ScrollView scrollView;
    public final TextView txt1Week;
    public final TextView txt24Hours;
    public final TextView txt3Days;
    public final FitHeightTextView txtConfirm;
    public final TextView txtDeviceLabel;
    public final TextView txtFilterAll;
    public final TextView txtFilterThis;
    public final TextView txtFilterTitle;
    public final TextView txtMaxSpeed;
    public final TextView txtNameLabel;
    public final TextView txtPlayCountLabel;
    public final TextView txtReleaseTimeLabel;
    public final FitHeightTextView txtReset;
    public final TextView txtSortPlayCount;
    public final TextView txtSortTime;
    public final TextView txtSortTitle;
    public final TextView txtSpeedLabel;
    public final TextView txtTimeLabel;
    public final View viewFilterTitleBg;
    public final View viewSortTitleBg;
    public final View viewWidth;

    private DialogFilterModeBinding(ConstraintLayout constraintLayout, EditText editText, EditText editText2, EditText editText3, EditText editText4, EditText editText5, ImageView imageView, ImageView imageView2, ImageView imageView3, LinearLayout linearLayout, ConstraintLayout constraintLayout2, FlexboxLayout flexboxLayout, FlexboxLayout flexboxLayout2, ConstraintLayout constraintLayout3, ConstraintLayout constraintLayout4, Guideline guideline, ProgressBar progressBar, ScrollView scrollView, TextView textView, TextView textView2, TextView textView3, FitHeightTextView fitHeightTextView, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10, TextView textView11, FitHeightTextView fitHeightTextView2, TextView textView12, TextView textView13, TextView textView14, TextView textView15, TextView textView16, View view, View view2, View view3) {
        this.rootView = constraintLayout;
        this.editEndTime = editText;
        this.editName = editText2;
        this.editPlayCountEnd = editText3;
        this.editPlayCountStart = editText4;
        this.editStartTime = editText5;
        this.imgMaxSpeed = imageView;
        this.imgSortPlayCount = imageView2;
        this.imgSortTime = imageView3;
        this.layoutFilterOnly = linearLayout;
        this.layoutMaxSpeed = constraintLayout2;
        this.layoutReleaseTime = flexboxLayout;
        this.layoutSortBy = flexboxLayout2;
        this.layoutSortPlayCount = constraintLayout3;
        this.layoutSortTime = constraintLayout4;
        this.lineMaxStart = guideline;
        this.progressSpeed = progressBar;
        this.scrollView = scrollView;
        this.txt1Week = textView;
        this.txt24Hours = textView2;
        this.txt3Days = textView3;
        this.txtConfirm = fitHeightTextView;
        this.txtDeviceLabel = textView4;
        this.txtFilterAll = textView5;
        this.txtFilterThis = textView6;
        this.txtFilterTitle = textView7;
        this.txtMaxSpeed = textView8;
        this.txtNameLabel = textView9;
        this.txtPlayCountLabel = textView10;
        this.txtReleaseTimeLabel = textView11;
        this.txtReset = fitHeightTextView2;
        this.txtSortPlayCount = textView12;
        this.txtSortTime = textView13;
        this.txtSortTitle = textView14;
        this.txtSpeedLabel = textView15;
        this.txtTimeLabel = textView16;
        this.viewFilterTitleBg = view;
        this.viewSortTitleBg = view2;
        this.viewWidth = view3;
    }

    public static DialogFilterModeBinding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        View findChildViewById3;
        int i = R.id.edit_end_time;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.edit_name;
            EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText2 != null) {
                i = R.id.edit_play_count_end;
                EditText editText3 = (EditText) ViewBindings.findChildViewById(view, i);
                if (editText3 != null) {
                    i = R.id.edit_play_count_start;
                    EditText editText4 = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText4 != null) {
                        i = R.id.edit_start_time;
                        EditText editText5 = (EditText) ViewBindings.findChildViewById(view, i);
                        if (editText5 != null) {
                            i = R.id.img_max_speed;
                            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView != null) {
                                i = R.id.img_sort_play_count;
                                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView2 != null) {
                                    i = R.id.img_sort_time;
                                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView3 != null) {
                                        i = R.id.layout_filter_only;
                                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout != null) {
                                            i = R.id.layout_max_speed;
                                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                            if (constraintLayout != null) {
                                                i = R.id.layout_release_time;
                                                FlexboxLayout flexboxLayout = (FlexboxLayout) ViewBindings.findChildViewById(view, i);
                                                if (flexboxLayout != null) {
                                                    i = R.id.layout_sort_by;
                                                    FlexboxLayout flexboxLayout2 = (FlexboxLayout) ViewBindings.findChildViewById(view, i);
                                                    if (flexboxLayout2 != null) {
                                                        i = R.id.layout_sort_play_count;
                                                        ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                        if (constraintLayout2 != null) {
                                                            i = R.id.layout_sort_time;
                                                            ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                            if (constraintLayout3 != null) {
                                                                i = R.id.line_max_start;
                                                                Guideline guideline = (Guideline) ViewBindings.findChildViewById(view, i);
                                                                if (guideline != null) {
                                                                    i = R.id.progress_speed;
                                                                    ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                    if (progressBar != null) {
                                                                        i = R.id.scroll_view;
                                                                        ScrollView scrollView = (ScrollView) ViewBindings.findChildViewById(view, i);
                                                                        if (scrollView != null) {
                                                                            i = R.id.txt_1_week;
                                                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView != null) {
                                                                                i = R.id.txt_24_hours;
                                                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView2 != null) {
                                                                                    i = R.id.txt_3_days;
                                                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView3 != null) {
                                                                                        i = R.id.txt_confirm;
                                                                                        FitHeightTextView fitHeightTextView = (FitHeightTextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (fitHeightTextView != null) {
                                                                                            i = R.id.txt_device_label;
                                                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView4 != null) {
                                                                                                i = R.id.txt_filter_all;
                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView5 != null) {
                                                                                                    i = R.id.txt_filter_this;
                                                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView6 != null) {
                                                                                                        i = R.id.txt_filter_title;
                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView7 != null) {
                                                                                                            i = R.id.txt_max_speed;
                                                                                                            TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (textView8 != null) {
                                                                                                                i = R.id.txt_name_label;
                                                                                                                TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView9 != null) {
                                                                                                                    i = R.id.txt_play_count_label;
                                                                                                                    TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (textView10 != null) {
                                                                                                                        i = R.id.txt_release_time_label;
                                                                                                                        TextView textView11 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (textView11 != null) {
                                                                                                                            i = R.id.txt_reset;
                                                                                                                            FitHeightTextView fitHeightTextView2 = (FitHeightTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (fitHeightTextView2 != null) {
                                                                                                                                i = R.id.txt_sort_play_count;
                                                                                                                                TextView textView12 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (textView12 != null) {
                                                                                                                                    i = R.id.txt_sort_time;
                                                                                                                                    TextView textView13 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                    if (textView13 != null) {
                                                                                                                                        i = R.id.txt_sort_title;
                                                                                                                                        TextView textView14 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                        if (textView14 != null) {
                                                                                                                                            i = R.id.txt_speed_label;
                                                                                                                                            TextView textView15 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                            if (textView15 != null) {
                                                                                                                                                i = R.id.txt_time_label;
                                                                                                                                                TextView textView16 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                if (textView16 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_filter_title_bg))) != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.view_sort_title_bg))) != null && (findChildViewById3 = ViewBindings.findChildViewById(view, (i = R.id.view_width))) != null) {
                                                                                                                                                    return new DialogFilterModeBinding((ConstraintLayout) view, editText, editText2, editText3, editText4, editText5, imageView, imageView2, imageView3, linearLayout, constraintLayout, flexboxLayout, flexboxLayout2, constraintLayout2, constraintLayout3, guideline, progressBar, scrollView, textView, textView2, textView3, fitHeightTextView, textView4, textView5, textView6, textView7, textView8, textView9, textView10, textView11, fitHeightTextView2, textView12, textView13, textView14, textView15, textView16, findChildViewById, findChildViewById2, findChildViewById3);
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

    public static DialogFilterModeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogFilterModeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_filter_mode, viewGroup, false);
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
