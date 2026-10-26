package com.hismith.hismithapp.pose;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import com.hismith.hismithapp.utils.LogUtil;
import java.io.IOException;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

/* loaded from: classes3.dex */
public class PoseLandmarkerHelper2 {
    private Future<?> currentTask;
    private VideoFrameCallback mCallback;
    private Context mContext;
    private Uri mVideoUri;
    private final ConcurrentSkipListMap<Long, PoseLandmarkerResult> resultCache = new ConcurrentSkipListMap<>();
    private long mInferenceIntervalMs = 100;
    private ExecutorService singleExecutor = Executors.newSingleThreadExecutor();
    private AtomicBoolean isCancelled = new AtomicBoolean(false);
    private final Object taskLock = new Object();

    public interface VideoFrameCallback {
        void onComplete(boolean z, int i, long j);

        void onFrameRatio(int i, int i2);
    }

    public PoseLandmarkerHelper2(Context context) {
        this.mContext = context;
    }

    private PoseLandmarker initPose(Context context) {
        try {
            return PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder().setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(Delegate.GPU).build()).setRunningMode(RunningMode.VIDEO).build());
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public void cancelAsyncTask() {
        Future<?> future = this.currentTask;
        if (future != null) {
            future.cancel(true);
        }
        this.isCancelled.set(true);
        this.resultCache.clear();
    }

    public void detectVideoFileAsync(final Uri uri, final long j, long j2, VideoFrameCallback videoFrameCallback) {
        synchronized (this.taskLock) {
            cancelAsyncTask();
            this.isCancelled.set(false);
            this.mVideoUri = uri;
            this.mInferenceIntervalMs = j2;
            this.mCallback = videoFrameCallback;
            this.currentTask = this.singleExecutor.submit(new Runnable() { // from class: com.hismith.hismithapp.pose.PoseLandmarkerHelper2$$ExternalSyntheticLambda0
                @Override // java.lang.Runnable
                public final void run() {
                    PoseLandmarkerHelper2.this.m616xad038196(uri, j);
                }
            });
        }
    }

    public void detectVideoFileAsync(Uri uri, long j, VideoFrameCallback videoFrameCallback) {
        detectVideoFileAsync(uri, 0L, j, videoFrameCallback);
    }

    public ConcurrentSkipListMap<Long, PoseLandmarkerResult> getResultCache() {
        return this.resultCache;
    }

    /* renamed from: lambda$detectVideoFileAsync$0$com-hismith-hismithapp-pose-PoseLandmarkerHelper2, reason: not valid java name */
    /* synthetic */ void m616xad038196(Uri uri, long j) {
        if (this.isCancelled.get() || Thread.currentThread().isInterrupted()) {
            return;
        }
        PoseLandmarker initPose = initPose(this.mContext);
        MediaMetadataRetriever mediaMetadataRetriever = new MediaMetadataRetriever();
        mediaMetadataRetriever.setDataSource(this.mContext, uri);
        String extractMetadata = mediaMetadataRetriever.extractMetadata(9);
        Long valueOf = extractMetadata != null ? Long.valueOf(Long.parseLong(extractMetadata)) : null;
        Bitmap frameAtTime = mediaMetadataRetriever.getFrameAtTime(0L);
        int width = frameAtTime != null ? frameAtTime.getWidth() : 0;
        int height = frameAtTime != null ? frameAtTime.getHeight() : 0;
        VideoFrameCallback videoFrameCallback = this.mCallback;
        if (videoFrameCallback != null) {
            videoFrameCallback.onFrameRatio(width, height);
        }
        if (valueOf == null || width == 0 || height == 0) {
            try {
                mediaMetadataRetriever.release();
                return;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        long longValue = valueOf.longValue();
        long j2 = this.mInferenceIntervalMs;
        long j3 = longValue / j2;
        for (long j4 = j / j2; j4 <= j3; j4++) {
            if (this.isCancelled.get() || Thread.currentThread().isInterrupted()) {
                try {
                    mediaMetadataRetriever.release();
                    break;
                } catch (IOException e2) {
                    throw new RuntimeException(e2);
                }
            }
            long j5 = this.mInferenceIntervalMs * j4;
            Bitmap frameAtTime2 = mediaMetadataRetriever.getFrameAtTime(1000 * j5, 3);
            if (frameAtTime2 != null) {
                if (frameAtTime2.getConfig() != Bitmap.Config.ARGB_8888) {
                    frameAtTime2 = frameAtTime2.copy(Bitmap.Config.ARGB_8888, false);
                }
                PoseLandmarkerResult detectForVideo = initPose != null ? initPose.detectForVideo(new BitmapImageBuilder(frameAtTime2).build(), j5) : null;
                if (detectForVideo != null) {
                    this.resultCache.put(Long.valueOf(j5), detectForVideo);
                    LogUtil.e("resultCache size:" + this.resultCache.size() + ", i:" + j4 + ", numberOfFrameToRead:" + j3);
                }
            }
        }
        try {
            mediaMetadataRetriever.release();
        } catch (IOException e3) {
            throw new RuntimeException(e3);
        }
    }

    public void seekToDetectVideoAsync(long j) {
        Uri uri = this.mVideoUri;
        if (uri == null) {
            return;
        }
        detectVideoFileAsync(uri, j, this.mInferenceIntervalMs, this.mCallback);
    }
}
