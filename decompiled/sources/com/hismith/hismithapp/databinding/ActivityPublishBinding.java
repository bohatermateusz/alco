package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.PictureAndTextEditorView;

/* loaded from: classes3.dex */
public final class ActivityPublishBinding implements ViewBinding {
    public final PictureAndTextEditorView publishEditContent;
    public final EditText publishEditTitle;
    public final ImageView publishImgAlbum;
    public final ImageView publishImgCamera;
    public final LinearLayout publishLayoutAction;
    public final LinearLayout publishLayoutEdit;
    public final ProgressBar publishProgress;
    public final ProgressBar publishTitleProgress;
    public final TextView publishTxtComplete;
    public final TextView publishTxtPublish;
    private final LinearLayout rootView;

    private ActivityPublishBinding(LinearLayout linearLayout, PictureAndTextEditorView pictureAndTextEditorView, EditText editText, ImageView imageView, ImageView imageView2, LinearLayout linearLayout2, LinearLayout linearLayout3, ProgressBar progressBar, ProgressBar progressBar2, TextView textView, TextView textView2) {
        this.rootView = linearLayout;
        this.publishEditContent = pictureAndTextEditorView;
        this.publishEditTitle = editText;
        this.publishImgAlbum = imageView;
        this.publishImgCamera = imageView2;
        this.publishLayoutAction = linearLayout2;
        this.publishLayoutEdit = linearLayout3;
        this.publishProgress = progressBar;
        this.publishTitleProgress = progressBar2;
        this.publishTxtComplete = textView;
        this.publishTxtPublish = textView2;
    }

    public static ActivityPublishBinding bind(View view) {
        int i = R.id.publish_edit_content;
        PictureAndTextEditorView pictureAndTextEditorView = (PictureAndTextEditorView) ViewBindings.findChildViewById(view, i);
        if (pictureAndTextEditorView != null) {
            i = R.id.publish_edit_title;
            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText != null) {
                i = R.id.publish_img_album;
                ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView != null) {
                    i = R.id.publish_img_camera;
                    ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView2 != null) {
                        i = R.id.publish_layout_action;
                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout != null) {
                            i = R.id.publish_layout_edit;
                            LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout2 != null) {
                                i = R.id.publish_progress;
                                ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                if (progressBar != null) {
                                    i = R.id.publish_title_progress;
                                    ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                    if (progressBar2 != null) {
                                        i = R.id.publish_txt_complete;
                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView != null) {
                                            i = R.id.publish_txt_publish;
                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView2 != null) {
                                                return new ActivityPublishBinding((LinearLayout) view, pictureAndTextEditorView, editText, imageView, imageView2, linearLayout, linearLayout2, progressBar, progressBar2, textView, textView2);
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

    public static ActivityPublishBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPublishBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_publish, viewGroup, false);
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
