package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ItemPiuTimerBinding implements ViewBinding {
    public final EditText editTime;
    public final ImageView imgDelete;
    public final ImageView imgPlayBg;
    public final ConstraintLayout layoutContent;
    public final ConstraintLayout layoutEdit;
    public final ConstraintLayout layoutItem;
    public final ConstraintLayout layoutTitle;
    private final ConstraintLayout rootView;
    public final TextView txtContentTitle;
    public final TextView txtTitle;

    private ItemPiuTimerBinding(ConstraintLayout constraintLayout, EditText editText, ImageView imageView, ImageView imageView2, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, ConstraintLayout constraintLayout4, ConstraintLayout constraintLayout5, TextView textView, TextView textView2) {
        this.rootView = constraintLayout;
        this.editTime = editText;
        this.imgDelete = imageView;
        this.imgPlayBg = imageView2;
        this.layoutContent = constraintLayout2;
        this.layoutEdit = constraintLayout3;
        this.layoutItem = constraintLayout4;
        this.layoutTitle = constraintLayout5;
        this.txtContentTitle = textView;
        this.txtTitle = textView2;
    }

    public static ItemPiuTimerBinding bind(View view) {
        int i = R.id.edit_time;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.img_delete;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_play_bg;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.layout_content;
                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                    if (constraintLayout != null) {
                        i = R.id.layout_edit;
                        ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                        if (constraintLayout2 != null) {
                            i = R.id.layout_item;
                            ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout3 != null) {
                                i = R.id.layout_title;
                                ConstraintLayout constraintLayout4 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                if (constraintLayout4 != null) {
                                    i = R.id.txt_content_title;
                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView != null) {
                                        i = R.id.txt_title;
                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView2 != null) {
                                            return new ItemPiuTimerBinding((ConstraintLayout) view, editText, imageView, imageView2, constraintLayout, constraintLayout2, constraintLayout3, constraintLayout4, textView, textView2);
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

    public static ItemPiuTimerBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ItemPiuTimerBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.item_piu_timer, viewGroup, false);
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
