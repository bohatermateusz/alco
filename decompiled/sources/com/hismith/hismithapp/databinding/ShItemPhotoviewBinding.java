package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.image_preview.view.helper.FingerDragHelper;
import com.hismith.hismithapp.image_preview.view.helper.SubsamplingScaleImageViewDragClose;
import com.hismith.hismithapp.image_preview.view.photoview.PhotoView;

/* loaded from: classes3.dex */
public final class ShItemPhotoviewBinding implements ViewBinding {
    public final FingerDragHelper fingerDragHelper;
    public final PhotoView gifView;
    public final SubsamplingScaleImageViewDragClose photoView;
    public final ProgressBar progressView;
    private final FrameLayout rootView;

    private ShItemPhotoviewBinding(FrameLayout frameLayout, FingerDragHelper fingerDragHelper, PhotoView photoView, SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose, ProgressBar progressBar) {
        this.rootView = frameLayout;
        this.fingerDragHelper = fingerDragHelper;
        this.gifView = photoView;
        this.photoView = subsamplingScaleImageViewDragClose;
        this.progressView = progressBar;
    }

    public static ShItemPhotoviewBinding bind(View view) {
        int i = R.id.fingerDragHelper;
        FingerDragHelper fingerDragHelper = (FingerDragHelper) ViewBindings.findChildViewById(view, i);
        if (fingerDragHelper != null) {
            i = R.id.gif_view;
            PhotoView photoView = (PhotoView) ViewBindings.findChildViewById(view, i);
            if (photoView != null) {
                i = R.id.photo_view;
                SubsamplingScaleImageViewDragClose subsamplingScaleImageViewDragClose = (SubsamplingScaleImageViewDragClose) ViewBindings.findChildViewById(view, i);
                if (subsamplingScaleImageViewDragClose != null) {
                    i = R.id.progress_view;
                    ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                    if (progressBar != null) {
                        return new ShItemPhotoviewBinding((FrameLayout) view, fingerDragHelper, photoView, subsamplingScaleImageViewDragClose, progressBar);
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ShItemPhotoviewBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ShItemPhotoviewBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.sh_item_photoview, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public FrameLayout getRoot() {
        return this.rootView;
    }
}
