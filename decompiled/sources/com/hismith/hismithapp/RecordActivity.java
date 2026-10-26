package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.SurfaceView;
import com.hismith.hismithapp.view.CaptureButton;
import com.hismith.hismithapp.view.RecordView;
import java.util.concurrent.atomic.AtomicBoolean;

/* loaded from: classes3.dex */
public class RecordActivity extends BaseActivity {
    private CaptureButton captureButton;
    private AtomicBoolean isShotPhoto = new AtomicBoolean(false);
    private RecordView recordView;
    private SurfaceView surfaceView;

    private void init() {
    }

    private void initMediaRecorder() {
    }

    private void initView() {
        this.surfaceView = (SurfaceView) findViewById(R.id.surfaceView);
        this.captureButton = (CaptureButton) findViewById(R.id.btn_capture);
        this.recordView = (RecordView) findViewById(R.id.btn_record);
    }

    private void setListener() {
        this.recordView.setOnGestureListener(new RecordView.OnGestureListener() { // from class: com.hismith.hismithapp.RecordActivity.1
            @Override // com.hismith.hismithapp.view.RecordView.OnGestureListener
            public void onClick() {
                RecordActivity.this.isShotPhoto.set(true);
            }

            @Override // com.hismith.hismithapp.view.RecordView.OnGestureListener
            public void onDown() {
            }

            @Override // com.hismith.hismithapp.view.RecordView.OnGestureListener
            public void onUp() {
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_record);
        initView();
        init();
        setListener();
        initMediaRecorder();
    }
}
