package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityEditBioBinding implements ViewBinding {
    public final EditText bioEditContent;
    public final ProgressBar bioTitleProgress;
    public final TextView bioTxtCount;
    public final TextView bioTxtPost;
    private final LinearLayout rootView;

    private ActivityEditBioBinding(LinearLayout linearLayout, EditText editText, ProgressBar progressBar, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.bioEditContent = editText;
        this.bioTitleProgress = progressBar;
        this.bioTxtCount = textView;
        this.bioTxtPost = textView2;
    }

    public static ActivityEditBioBinding bind(View view) {
        int i = R.id.bio_edit_content;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.bio_title_progress;
            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
            if (progressBar != null) {
                i = R.id.bio_txt_count;
                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView != null) {
                    i = R.id.bio_txt_post;
                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView2 != null) {
                        return new ActivityEditBioBinding((LinearLayout) view, editText, progressBar, textView, textView2);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityEditBioBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityEditBioBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_edit_bio, viewGroup, false);
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
