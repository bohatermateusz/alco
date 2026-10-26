package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogReportSelectBinding implements ViewBinding {
    public final EditText reportEditContent;
    public final LinearLayout reportLayoutContent;
    public final Button reportSelectBtnSubmit;
    public final RecyclerView reportSelectRecyclerView;
    public final ScrollView reportSelectScroll;
    public final TextView reportSelectTxtTitle;
    public final TextView reportTxtCount;
    public final TextView reportTxtLabel;
    private final ConstraintLayout rootView;

    private DialogReportSelectBinding(ConstraintLayout constraintLayout, EditText editText, LinearLayout linearLayout, Button button, RecyclerView recyclerView, ScrollView scrollView, TextView textView, TextView textView2, TextView textView3) {
        this.rootView = constraintLayout;
        this.reportEditContent = editText;
        this.reportLayoutContent = linearLayout;
        this.reportSelectBtnSubmit = button;
        this.reportSelectRecyclerView = recyclerView;
        this.reportSelectScroll = scrollView;
        this.reportSelectTxtTitle = textView;
        this.reportTxtCount = textView2;
        this.reportTxtLabel = textView3;
    }

    public static DialogReportSelectBinding bind(View view) {
        int i = R.id.report_edit_content;
        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
        if (editText != null) {
            i = R.id.report_layout_content;
            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
            if (linearLayout != null) {
                i = R.id.report_select_btn_submit;
                Button button = (Button) ViewBindings.findChildViewById(view, i);
                if (button != null) {
                    i = R.id.report_select_recyclerView;
                    RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                    if (recyclerView != null) {
                        i = R.id.report_select_scroll;
                        ScrollView scrollView = (ScrollView) ViewBindings.findChildViewById(view, i);
                        if (scrollView != null) {
                            i = R.id.report_select_txt_title;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.report_txt_count;
                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView2 != null) {
                                    i = R.id.report_txt_label;
                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView3 != null) {
                                        return new DialogReportSelectBinding((ConstraintLayout) view, editText, linearLayout, button, recyclerView, scrollView, textView, textView2, textView3);
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

    public static DialogReportSelectBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogReportSelectBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_report_select, viewGroup, false);
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
