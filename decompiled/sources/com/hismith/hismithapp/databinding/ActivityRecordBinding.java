package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CaptureButton;
import com.hismith.hismithapp.view.RecordView;

/* loaded from: classes3.dex */
public final class ActivityRecordBinding implements ViewBinding {
    public final CaptureButton btnCapture;
    public final RecordView btnRecord;
    public final ConstraintLayout main;
    private final ConstraintLayout rootView;
    public final SurfaceView surfaceView;

    private ActivityRecordBinding(ConstraintLayout constraintLayout, CaptureButton captureButton, RecordView recordView, ConstraintLayout constraintLayout2, SurfaceView surfaceView) {
        this.rootView = constraintLayout;
        this.btnCapture = captureButton;
        this.btnRecord = recordView;
        this.main = constraintLayout2;
        this.surfaceView = surfaceView;
    }

    public static ActivityRecordBinding bind(View view) {
        int i = R.id.btn_capture;
        CaptureButton captureButton = (CaptureButton) ViewBindings.findChildViewById(view, i);
        if (captureButton != null) {
            i = R.id.btn_record;
            RecordView recordView = (RecordView) ViewBindings.findChildViewById(view, i);
            if (recordView != null) {
                ConstraintLayout constraintLayout = (ConstraintLayout) view;
                i = R.id.surfaceView;
                SurfaceView surfaceView = (SurfaceView) ViewBindings.findChildViewById(view, i);
                if (surfaceView != null) {
                    return new ActivityRecordBinding(constraintLayout, captureButton, recordView, constraintLayout, surfaceView);
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityRecordBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityRecordBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_record, viewGroup, false);
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
