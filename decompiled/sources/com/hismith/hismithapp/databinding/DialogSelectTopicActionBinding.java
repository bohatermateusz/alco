package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogSelectTopicActionBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final TextView topicActionCancel;
    public final TextView topicActionDelete;
    public final View topicActionDeleteLine;
    public final TextView topicActionNotInterested;
    public final TextView topicActionReport;

    private DialogSelectTopicActionBinding(LinearLayout linearLayout, TextView textView, TextView textView2, View view, TextView textView3, TextView textView4) {
        this.rootView = linearLayout;
        this.topicActionCancel = textView;
        this.topicActionDelete = textView2;
        this.topicActionDeleteLine = view;
        this.topicActionNotInterested = textView3;
        this.topicActionReport = textView4;
    }

    public static DialogSelectTopicActionBinding bind(View view) {
        View findChildViewById;
        int i = R.id.topic_action_cancel;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.topic_action_delete;
            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView2 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.topic_action_delete_line))) != null) {
                i = R.id.topic_action_notInterested;
                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView3 != null) {
                    i = R.id.topic_action_report;
                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView4 != null) {
                        return new DialogSelectTopicActionBinding((LinearLayout) view, textView, textView2, findChildViewById, textView3, textView4);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogSelectTopicActionBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogSelectTopicActionBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_select_topic_action, viewGroup, false);
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
