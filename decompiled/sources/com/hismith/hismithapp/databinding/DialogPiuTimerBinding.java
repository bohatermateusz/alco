package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Group;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogPiuTimerBinding implements ViewBinding {
    public final EditText editMaxTime;
    public final EditText editMinTime;
    public final Group groupList;
    public final ImageView imgIcPause;
    public final ImageView imgRandomPlayBg;
    public final ConstraintLayout layoutContent;
    public final ConstraintLayout layoutItem;
    public final LinearLayout layoutPause;
    public final ConstraintLayout layoutRandom;
    public final ConstraintLayout layoutRandomTitle;
    public final LinearLayout layoutSelected;
    public final View layoutTitle;
    public final RecyclerView recyclerTimer;
    private final ConstraintLayout rootView;
    public final ImageView titleImgBack;
    public final TextView txtAdd;
    public final TextView txtOne;
    public final TextView txtPause;
    public final TextView txtRandom;
    public final TextView txtRandomTitle;
    public final TextView txtThree;
    public final TextView txtTimeSeparator;
    public final TextView txtTitle;
    public final TextView txtTwo;

    private DialogPiuTimerBinding(ConstraintLayout constraintLayout, EditText editText, EditText editText2, Group group, ImageView imageView, ImageView imageView2, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, LinearLayout linearLayout, ConstraintLayout constraintLayout4, ConstraintLayout constraintLayout5, LinearLayout linearLayout2, View view, RecyclerView recyclerView, ImageView imageView3, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9) {
        this.rootView = constraintLayout;
        this.editMaxTime = editText;
        this.editMinTime = editText2;
        this.groupList = group;
        this.imgIcPause = imageView;
        this.imgRandomPlayBg = imageView2;
        this.layoutContent = constraintLayout2;
        this.layoutItem = constraintLayout3;
        this.layoutPause = linearLayout;
        this.layoutRandom = constraintLayout4;
        this.layoutRandomTitle = constraintLayout5;
        this.layoutSelected = linearLayout2;
        this.layoutTitle = view;
        this.recyclerTimer = recyclerView;
        this.titleImgBack = imageView3;
        this.txtAdd = textView;
        this.txtOne = textView2;
        this.txtPause = textView3;
        this.txtRandom = textView4;
        this.txtRandomTitle = textView5;
        this.txtThree = textView6;
        this.txtTimeSeparator = textView7;
        this.txtTitle = textView8;
        this.txtTwo = textView9;
    }

    public static DialogPiuTimerBinding bind(View view) {
        View findChildViewById;
        int i = R.id.edit_max_time;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.edit_min_time;
            EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText2 != null) {
                i = R.id.group_list;
                Group group = (Group) ViewBindings.findChildViewById(view, i);
                if (group != null) {
                    i = R.id.img_ic_pause;
                    ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView != null) {
                        i = R.id.img_random_play_bg;
                        ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView2 != null) {
                            i = R.id.layout_content;
                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout != null) {
                                i = R.id.layout_item;
                                ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                if (constraintLayout2 != null) {
                                    i = R.id.layout_pause;
                                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout != null) {
                                        i = R.id.layout_random;
                                        ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                        if (constraintLayout3 != null) {
                                            i = R.id.layout_random_title;
                                            ConstraintLayout constraintLayout4 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                            if (constraintLayout4 != null) {
                                                i = R.id.layout_selected;
                                                LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                if (linearLayout2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.layout_title))) != null) {
                                                    i = R.id.recycler_timer;
                                                    RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                    if (recyclerView != null) {
                                                        i = R.id.title_img_back;
                                                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                        if (imageView3 != null) {
                                                            i = R.id.txt_add;
                                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView != null) {
                                                                i = R.id.txt_one;
                                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView2 != null) {
                                                                    i = R.id.txt_pause;
                                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView3 != null) {
                                                                        i = R.id.txt_random;
                                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView4 != null) {
                                                                            i = R.id.txt_random_title;
                                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView5 != null) {
                                                                                i = R.id.txt_three;
                                                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView6 != null) {
                                                                                    i = R.id.txt_time_separator;
                                                                                    TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView7 != null) {
                                                                                        i = R.id.txt_title;
                                                                                        TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView8 != null) {
                                                                                            i = R.id.txt_two;
                                                                                            TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView9 != null) {
                                                                                                return new DialogPiuTimerBinding((ConstraintLayout) view, editText, editText2, group, imageView, imageView2, constraintLayout, constraintLayout2, linearLayout, constraintLayout3, constraintLayout4, linearLayout2, findChildViewById, recyclerView, imageView3, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9);
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

    public static DialogPiuTimerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogPiuTimerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_piu_timer, viewGroup, false);
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
