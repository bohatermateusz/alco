package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatCheckBox;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.GuideButton;

/* loaded from: classes3.dex */
public final class ViewFeatureGuideBinding implements ViewBinding {
    public final TextView btnFeatureGuideGotIt;
    public final AppCompatCheckBox chkFeatureGuideNoMore;
    public final ConstraintLayout layoutFeatureGuide;
    private final ConstraintLayout rootView;
    public final TextView txtFeatureGuideContent;
    public final TextView txtFeatureGuideDesc;
    public final GuideButton txtFeatureGuideLabel;
    public final AppCompatTextView txtFeatureGuideTitle;

    private ViewFeatureGuideBinding(ConstraintLayout constraintLayout, TextView textView, AppCompatCheckBox appCompatCheckBox, ConstraintLayout constraintLayout2, TextView textView2, TextView textView3, GuideButton guideButton, AppCompatTextView appCompatTextView) {
        this.rootView = constraintLayout;
        this.btnFeatureGuideGotIt = textView;
        this.chkFeatureGuideNoMore = appCompatCheckBox;
        this.layoutFeatureGuide = constraintLayout2;
        this.txtFeatureGuideContent = textView2;
        this.txtFeatureGuideDesc = textView3;
        this.txtFeatureGuideLabel = guideButton;
        this.txtFeatureGuideTitle = appCompatTextView;
    }

    public static ViewFeatureGuideBinding bind(View view) {
        int i = R.id.btn_feature_guide_got_it;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.chk_feature_guide_no_more;
            AppCompatCheckBox appCompatCheckBox = (AppCompatCheckBox) ViewBindings.findChildViewById(view, i);
            if (appCompatCheckBox != null) {
                ConstraintLayout constraintLayout = (ConstraintLayout) view;
                i = R.id.txt_feature_guide_content;
                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView2 != null) {
                    i = R.id.txt_feature_guide_desc;
                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                    if (textView3 != null) {
                        i = R.id.txt_feature_guide_label;
                        GuideButton guideButton = (GuideButton) ViewBindings.findChildViewById(view, i);
                        if (guideButton != null) {
                            i = R.id.txt_feature_guide_title;
                            AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                            if (appCompatTextView != null) {
                                return new ViewFeatureGuideBinding(constraintLayout, textView, appCompatCheckBox, constraintLayout, textView2, textView3, guideButton, appCompatTextView);
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ViewFeatureGuideBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ViewFeatureGuideBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.view_feature_guide, viewGroup, false);
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
