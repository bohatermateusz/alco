package com.hismith.hismithapp.pose;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.MediaCodec;
import android.media.MediaCrypto;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.view.Surface;
import androidx.media3.exoplayer.Renderer;
import com.facebook.appevents.internal.ViewHierarchyConstants;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import com.hismith.hismithapp.utils.LogUtil;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/* loaded from: classes3.dex */
public class PoseLandmarkerHelper4 {
    private static final long DEQUEUE_TIMEOUT_US = 1000;
    public static final int ERROR_DECODE_EXCEPTION = 4;
    public static final int ERROR_INVALID_VIDEO_META = 3;
    public static final int ERROR_MODEL_INIT_FAILED = 1;
    public static final int ERROR_NONE = 0;
    public static final int ERROR_NO_VIDEO_TRACK = 2;
    private static final long SEEK_DEBOUNCE_DELAY_MS = 300;
    private static final String TAG = "PoseLandmarkerHelper4";
    private Future<?> currentTask;
    private VideoFrameCallback mCallback;
    private final Context mContext;
    private Uri mCurrentProcessingUri;
    private PoseLandmarker mPoseLandmarker;
    private final ConcurrentSkipListMap<Long, PoseLandmarkerResult> resultCache = new ConcurrentSkipListMap<>();
    private long mInferenceIntervalMs = 100;
    private long mPastKeepMs = Renderer.DEFAULT_DURATION_TO_PROGRESS_US;
    private long mFutureMaxMs = 50000;
    private long mPlayReadyMs = Renderer.DEFAULT_DURATION_TO_PROGRESS_US;
    private long mSeekBackMs = 5000;
    private ExecutorService singleExecutor = Executors.newSingleThreadExecutor();
    private final AtomicInteger taskGeneration = new AtomicInteger(0);
    private final Object taskLock = new Object();
    private long mLastPlayTime = 0;
    private volatile long mCurrentPlayTime = 0;
    private volatile boolean isReadyCallback = false;
    private volatile boolean isIntact = false;
    private long mVideoDurationMs = -1;
    private int mVideoWidth = 0;
    private int mVideoHeight = 0;
    private long mTotalFrames = -1;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private final Runnable mSeekRunnable = new Runnable() { // from class: com.hismith.hismithapp.pose.PoseLandmarkerHelper4.1
        @Override // java.lang.Runnable
        public void run() {
            PoseLandmarkerHelper4 poseLandmarkerHelper4 = PoseLandmarkerHelper4.this;
            poseLandmarkerHelper4.doSeekInternal(poseLandmarkerHelper4.mPendingPlayTime);
        }
    };
    private long mPendingPlayTime = -1;

    public interface VideoFrameCallback {
        void onCacheProgress(int i, long j);

        void onCanPlayed();

        void onComplete(boolean z, int i, long j, int i2, String str);

        void onFrameRatio(int i, int i2);
    }

    public PoseLandmarkerHelper4(Context context) {
        this.mContext = context;
    }

    private void cancelAsyncTask(boolean z) {
        this.taskGeneration.incrementAndGet();
        Future<?> future = this.currentTask;
        if (future != null) {
            future.cancel(true);
            this.currentTask = null;
        }
        if (z) {
            this.resultCache.clear();
            this.mCurrentProcessingUri = null;
            this.mVideoDurationMs = -1L;
            this.mVideoWidth = 0;
            this.mVideoHeight = 0;
            this.mTotalFrames = -1L;
            this.mCurrentPlayTime = 0L;
            this.isReadyCallback = false;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void doSeekInternal(long j) {
        synchronized (this.taskLock) {
            if (this.mCurrentProcessingUri == null) {
                return;
            }
            startDetectionTask(j);
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:15:0x000b, code lost:
    
        if (r3 <= r2.mLastPlayTime) goto L7;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private synchronized PoseLandmarker getOrCreatePoseLandmarker(long j) {
        if (j != 0) {
        }
        this.mPoseLandmarker = initPose(this.mContext);
        return this.mPoseLandmarker;
    }

    private PoseLandmarker initPose(Context context) {
        try {
            return PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder().setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(Delegate.GPU).build()).setNumPoses(1).setRunningMode(RunningMode.VIDEO).build());
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private boolean isReadyAt(long j) {
        Long lastKey = this.resultCache.lastKey();
        if (lastKey == null) {
            return false;
        }
        if (lastKey.longValue() - j >= this.mPlayReadyMs) {
            return true;
        }
        return this.mVideoDurationMs > 0 && lastKey.longValue() >= this.mVideoDurationMs - this.mInferenceIntervalMs;
    }

    private boolean isSameUri(Uri uri, Uri uri2) {
        if (uri == null && uri2 == null) {
            return true;
        }
        if (uri == null || uri2 == null) {
            return false;
        }
        return uri.normalizeScheme().toString().equals(uri2.normalizeScheme().toString());
    }

    private void notifyComplete(int i, boolean z, int i2, int i3, String str) {
        if (i != this.taskGeneration.get() || this.mCallback == null) {
            return;
        }
        if (!this.isReadyCallback) {
            this.mCallback.onCanPlayed();
            this.isReadyCallback = true;
        }
        LogUtil.d("PoseLandmarkerHelper4 onComplete success=" + z + ", frames=" + i2 + ", errorCode=" + i3 + ", errorMsg=" + str);
        this.mCallback.onComplete(z, i2, 0L, i3, str);
    }

    private void processOutputFrame(MediaCodec mediaCodec, int i, MediaFormat mediaFormat, long j, long j2, long j3, PoseLandmarker poseLandmarker, int i2) {
        ByteBuffer outputBuffer = mediaCodec.getOutputBuffer(i);
        if (outputBuffer == null) {
            return;
        }
        Bitmap yuv420ToBitmap = yuv420ToBitmap(outputBuffer, this.mVideoWidth, this.mVideoHeight, mediaFormat.containsKey("color-format") ? mediaFormat.getInteger("color-format") : 21);
        if (yuv420ToBitmap == null) {
            return;
        }
        MPImage mPImage = null;
        try {
            try {
                mPImage = new BitmapImageBuilder(yuv420ToBitmap).build();
                PoseLandmarkerResult detectForVideo = poseLandmarker.detectForVideo(mPImage, j);
                if (detectForVideo != null && i2 == this.taskGeneration.get()) {
                    this.resultCache.put(Long.valueOf(j2), detectForVideo);
                    this.mLastPlayTime = j;
                    VideoFrameCallback videoFrameCallback = this.mCallback;
                    if (videoFrameCallback != null) {
                        videoFrameCallback.onCacheProgress(this.resultCache.size(), j2);
                    }
                    LogUtil.w("PoseLandmarkerHelper4 已缓存 size=" + this.resultCache.size() + ", key=" + j2 + ", duration=" + this.mVideoDurationMs);
                    if (this.mCallback != null && !this.isReadyCallback && isReadyAt(j3)) {
                        this.mCallback.onCanPlayed();
                        this.isReadyCallback = true;
                        LogUtil.d("PoseLandmarkerHelper4 onCanPlayed 触发，目标=" + j3 + ", 已缓存到=" + j2);
                    }
                }
                if (mPImage != null) {
                    try {
                        mPImage.close();
                    } catch (Exception unused) {
                    }
                }
                if (yuv420ToBitmap == null || yuv420ToBitmap.isRecycled()) {
                    return;
                }
            } finally {
            }
        } catch (Exception e) {
            e.printStackTrace();
            if (mPImage != null) {
                try {
                    mPImage.close();
                } catch (Exception unused2) {
                }
            }
            if (yuv420ToBitmap == null || yuv420ToBitmap.isRecycled()) {
                return;
            }
        }
        yuv420ToBitmap.recycle();
    }

    private void startDetectionTask(final long j) {
        if (this.singleExecutor.isShutdown()) {
            this.singleExecutor = Executors.newSingleThreadExecutor();
        }
        this.isReadyCallback = false;
        final int incrementAndGet = this.taskGeneration.incrementAndGet();
        this.currentTask = this.singleExecutor.submit(new Runnable() { // from class: com.hismith.hismithapp.pose.PoseLandmarkerHelper4$$ExternalSyntheticLambda0
            @Override // java.lang.Runnable
            public final void run() {
                PoseLandmarkerHelper4.this.m618xc03e2798(j, incrementAndGet);
            }
        });
    }

    private void trimCacheWindow(long j) {
        this.resultCache.headMap((ConcurrentSkipListMap<Long, PoseLandmarkerResult>) Long.valueOf(j - this.mPastKeepMs), false).clear();
        this.resultCache.tailMap((ConcurrentSkipListMap<Long, PoseLandmarkerResult>) Long.valueOf(j + this.mFutureMaxMs), false).clear();
    }

    private Bitmap yuv420ToBitmap(ByteBuffer byteBuffer, int i, int i2, int i3) {
        int i4 = i * i2;
        int i5 = i4 / 2;
        int i6 = i4 + i5;
        if (byteBuffer.remaining() < i6) {
            LogUtil.e("PoseLandmarkerHelper4 yuv420ToBitmap: buffer too small " + byteBuffer.remaining() + " < " + i6);
            return null;
        }
        byte[] bArr = new byte[i6];
        byteBuffer.position(0);
        byteBuffer.get(bArr, 0, i6);
        byte[] bArr2 = new byte[i6];
        System.arraycopy(bArr, 0, bArr2, 0, i4);
        if (i3 == 19) {
            int i7 = i5 / 2;
            int i8 = i4 + i7;
            int i9 = i4;
            for (int i10 = 0; i10 < i7; i10++) {
                int i11 = i9 + 1;
                bArr2[i9] = bArr[i8 + i10];
                i9 += 2;
                bArr2[i11] = bArr[i4 + i10];
            }
        } else if (i3 == 21) {
            for (int i12 = 0; i12 < i5; i12 += 2) {
                int i13 = i4 + i12;
                byte b = bArr[i13];
                int i14 = i13 + 1;
                bArr2[i13] = bArr[i14];
                bArr2[i14] = b;
            }
        } else {
            for (int i15 = 0; i15 < i5; i15 += 2) {
                int i16 = i4 + i15;
                byte b2 = bArr[i16];
                int i17 = i16 + 1;
                bArr2[i16] = bArr[i17];
                bArr2[i17] = b2;
            }
        }
        try {
            YuvImage yuvImage = new YuvImage(bArr2, 17, i, i2, null);
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, i, i2), 80, byteArrayOutputStream);
            byte[] byteArray = byteArrayOutputStream.toByteArray();
            byteArrayOutputStream.close();
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length, options);
        } catch (Exception e) {
            LogUtil.e("PoseLandmarkerHelper4 yuv420ToBitmap error: " + e.getMessage());
            return null;
        }
    }

    public void cancelAll() {
        this.mHandler.removeCallbacks(this.mSeekRunnable);
        cancelAsyncTask(true);
        try {
            ExecutorService executorService = this.singleExecutor;
            if (executorService == null || executorService.isShutdown()) {
                return;
            }
            this.singleExecutor.shutdownNow();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void detectVideoFileAsync(Uri uri, long j, VideoFrameCallback videoFrameCallback) {
        synchronized (this.taskLock) {
            this.mHandler.removeCallbacks(this.mSeekRunnable);
            if (isSameUri(this.mCurrentProcessingUri, uri)) {
                Future<?> future = this.currentTask;
                if ((future == null || future.isDone()) && !this.isIntact) {
                    LogUtil.d("PoseLandmarkerHelper4 detectVideoFileAsync: same video but cache incomplete, restart.");
                }
                LogUtil.d("PoseLandmarkerHelper4 detectVideoFileAsync: same video already loading/loaded, ignore.");
                return;
            }
            cancelAsyncTask(true);
            this.mCurrentProcessingUri = uri;
            this.mInferenceIntervalMs = j;
            this.mCallback = videoFrameCallback;
            this.isIntact = false;
            this.mVideoDurationMs = -1L;
            this.mVideoWidth = 0;
            this.mVideoHeight = 0;
            this.mTotalFrames = -1L;
            this.mCurrentPlayTime = 0L;
            LogUtil.d("PoseLandmarkerHelper4 detectVideoFileAsync: start from 0");
            startDetectionTask(0L);
        }
    }

    public PoseLandmarkerResult getCacheResult(long j) {
        Long validTime = getValidTime(j);
        if (validTime == null) {
            return null;
        }
        this.mCurrentPlayTime = validTime.longValue();
        trimCacheWindow(validTime.longValue());
        LogUtil.d("PoseLandmarkerHelper4 getCacheResult playTime=" + j + ", key=" + validTime + ", cacheSize=" + this.resultCache.size());
        return this.resultCache.get(validTime);
    }

    public ConcurrentSkipListMap<Long, PoseLandmarkerResult> getResultCache() {
        return this.resultCache;
    }

    public PoseLandmarkerResult getValidResult(long j) {
        Long validTime = getValidTime(j);
        if (validTime == null) {
            return null;
        }
        return this.resultCache.get(validTime);
    }

    public Long getValidTime(long j) {
        Long floorKey = this.resultCache.floorKey(Long.valueOf(j));
        if (floorKey != null && j - floorKey.longValue() <= this.mInferenceIntervalMs) {
            return floorKey;
        }
        return null;
    }

    /* JADX WARN: Code restructure failed: missing block: B:80:0x020f, code lost:
    
        if (r18 != false) goto L116;
     */
    /* JADX WARN: Removed duplicated region for block: B:100:0x02c2 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:112:0x02d7 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:118:? A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:119:0x02cf A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:152:0x01b6 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:65:0x01be A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:93:0x02cb A[ORIG_RETURN, RETURN] */
    /* renamed from: lambda$startDetectionTask$0$com-hismith-hismithapp-pose-PoseLandmarkerHelper4, reason: not valid java name */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    /* synthetic */ void m618xc03e2798(long j, int i) {
        MediaExtractor mediaExtractor;
        int i2;
        MediaCodec mediaCodec;
        long j2;
        int i3;
        boolean z;
        boolean z2;
        boolean z3;
        boolean z4;
        int dequeueOutputBuffer;
        MediaFormat mediaFormat;
        boolean z5;
        int i4;
        int dequeueInputBuffer;
        long j3 = j;
        PoseLandmarker orCreatePoseLandmarker = getOrCreatePoseLandmarker(j);
        if (orCreatePoseLandmarker == null) {
            notifyComplete(i, false, 0, 1, "模型初始化失败");
            return;
        }
        MediaCodec mediaCodec2 = null;
        try {
            mediaExtractor = new MediaExtractor();
            try {
                mediaExtractor.setDataSource(this.mContext, this.mCurrentProcessingUri, (Map<String, String>) null);
                i2 = 0;
                while (true) {
                    if (i2 >= mediaExtractor.getTrackCount()) {
                        i2 = -1;
                        break;
                    }
                    try {
                        try {
                            String string = mediaExtractor.getTrackFormat(i2).getString("mime");
                            if (string != null && string.startsWith("video/")) {
                                break;
                            } else {
                                i2++;
                            }
                        } catch (Throwable th) {
                            th = th;
                            if (mediaCodec2 != null) {
                                try {
                                    mediaCodec2.stop();
                                } catch (Exception unused) {
                                }
                                try {
                                    mediaCodec2.release();
                                } catch (Exception unused2) {
                                }
                            }
                            if (mediaExtractor == null) {
                                throw th;
                            }
                            try {
                                mediaExtractor.release();
                                throw th;
                            } catch (Exception unused3) {
                                throw th;
                            }
                        }
                    } catch (Exception e) {
                        e = e;
                        e.printStackTrace();
                        LogUtil.e("error:" + e.getMessage());
                        notifyComplete(i, false, 0, 4, e.getMessage());
                        if (mediaCodec2 != null) {
                        }
                        if (mediaExtractor == null) {
                        }
                        try {
                            mediaExtractor.release();
                            return;
                        } catch (Exception unused4) {
                            return;
                        }
                    }
                }
            } catch (Exception e2) {
                e = e2;
            } catch (Throwable th2) {
                th = th2;
            }
        } catch (Exception e3) {
            e = e3;
            mediaExtractor = null;
        } catch (Throwable th3) {
            th = th3;
            mediaExtractor = null;
        }
        if (i2 < 0) {
            try {
                LogUtil.e("PoseLandmarkerHelper4 startDetectionTask: no video track");
                try {
                    notifyComplete(i, false, 0, 2, "未找到视频轨道");
                } catch (Exception e4) {
                    e = e4;
                    e.printStackTrace();
                    LogUtil.e("error:" + e.getMessage());
                    notifyComplete(i, false, 0, 4, e.getMessage());
                    if (mediaCodec2 != null) {
                        try {
                            mediaCodec2.stop();
                        } catch (Exception unused5) {
                        }
                        try {
                            mediaCodec2.release();
                        } catch (Exception unused6) {
                        }
                    }
                    if (mediaExtractor == null) {
                        return;
                    }
                    mediaExtractor.release();
                    return;
                }
            } catch (Exception e5) {
                e = e5;
            }
        } else {
            int i5 = i;
            mediaExtractor.selectTrack(i2);
            MediaFormat trackFormat = mediaExtractor.getTrackFormat(i2);
            this.mVideoWidth = trackFormat.getInteger(ViewHierarchyConstants.DIMENSION_WIDTH_KEY);
            this.mVideoHeight = trackFormat.getInteger(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY);
            try {
                long j4 = (trackFormat.containsKey("durationUs") ? trackFormat.getLong("durationUs") : -1L) / 1000;
                this.mVideoDurationMs = j4;
                j2 = 1000;
                this.mTotalFrames = j4 / this.mInferenceIntervalMs;
                VideoFrameCallback videoFrameCallback = this.mCallback;
                if (videoFrameCallback != null) {
                    try {
                        videoFrameCallback.onFrameRatio(this.mVideoWidth, this.mVideoHeight);
                    } catch (Exception e6) {
                        e = e6;
                        mediaCodec2 = null;
                        e.printStackTrace();
                        LogUtil.e("error:" + e.getMessage());
                        notifyComplete(i, false, 0, 4, e.getMessage());
                        if (mediaCodec2 != null) {
                        }
                        if (mediaExtractor == null) {
                        }
                        mediaExtractor.release();
                        return;
                    } catch (Throwable th4) {
                        th = th4;
                        mediaCodec2 = null;
                        if (mediaCodec2 != null) {
                        }
                        if (mediaExtractor == null) {
                        }
                    }
                }
            } catch (Exception e7) {
                e = e7;
                mediaCodec = null;
            } catch (Throwable th5) {
                th = th5;
                mediaCodec = null;
            }
            if (this.mVideoDurationMs > 0 && this.mVideoWidth != 0 && this.mVideoHeight != 0) {
                MediaCodec createDecoderByType = MediaCodec.createDecoderByType(trackFormat.getString("mime"));
                try {
                    try {
                        createDecoderByType.configure(trackFormat, (Surface) null, (MediaCrypto) null, 0);
                        createDecoderByType.start();
                        long max = Math.max((j3 - this.mSeekBackMs) * 1000, 0L);
                        LogUtil.d("PoseLandmarkerHelper4 跳转解码：目标=" + j3 + "ms, 实际seek=" + max + "us");
                        mediaExtractor.seekTo(max, 2);
                        MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
                        boolean z6 = false;
                        boolean z7 = false;
                        while (!z7) {
                            try {
                                if (i5 != this.taskGeneration.get()) {
                                    break;
                                }
                                try {
                                    if (!this.resultCache.isEmpty()) {
                                        Long lastKey = this.resultCache.lastKey();
                                        long max2 = Math.max(this.mCurrentPlayTime, j3);
                                        if (lastKey != null) {
                                            z2 = z7;
                                            if (lastKey.longValue() - max2 > this.mFutureMaxMs) {
                                                try {
                                                    LogUtil.d("PoseLandmarkerHelper4 缓存超前，休眠等待播放推进，lastKey=" + lastKey + ", consumer=" + max2);
                                                    Thread.sleep(50L);
                                                    z7 = z2;
                                                } catch (InterruptedException unused7) {
                                                }
                                            }
                                            if (!z6 || (dequeueInputBuffer = createDecoderByType.dequeueInputBuffer(j2)) < 0) {
                                                mediaCodec2 = createDecoderByType;
                                                z3 = false;
                                            } else {
                                                z3 = false;
                                                int readSampleData = mediaExtractor.readSampleData(createDecoderByType.getInputBuffer(dequeueInputBuffer), 0);
                                                if (readSampleData < 0) {
                                                    MediaCodec mediaCodec3 = createDecoderByType;
                                                    try {
                                                        mediaCodec3.queueInputBuffer(dequeueInputBuffer, 0, 0, 0L, 4);
                                                        mediaCodec2 = mediaCodec3;
                                                        z4 = true;
                                                        dequeueOutputBuffer = mediaCodec2.dequeueOutputBuffer(bufferInfo, 1000L);
                                                        if (dequeueOutputBuffer == -2) {
                                                            createDecoderByType = mediaCodec2;
                                                            z6 = z4;
                                                            z7 = z2;
                                                            j2 = 1000;
                                                        } else {
                                                            if (dequeueOutputBuffer >= 0) {
                                                                boolean z8 = (bufferInfo.flags & 4) != 0 ? true : z3;
                                                                if (z8) {
                                                                    mediaFormat = trackFormat;
                                                                    i4 = dequeueOutputBuffer;
                                                                    i3 = i5;
                                                                    z5 = true;
                                                                    j2 = 1000;
                                                                } else {
                                                                    j2 = 1000;
                                                                    long j5 = bufferInfo.presentationTimeUs / 1000;
                                                                    long j6 = this.mInferenceIntervalMs;
                                                                    long j7 = (j5 / j6) * j6;
                                                                    if (j5 < j3 || this.resultCache.containsKey(Long.valueOf(j7))) {
                                                                        mediaFormat = trackFormat;
                                                                        i3 = i5;
                                                                        i4 = dequeueOutputBuffer;
                                                                        z5 = true;
                                                                    } else {
                                                                        i4 = dequeueOutputBuffer;
                                                                        z5 = true;
                                                                        mediaFormat = trackFormat;
                                                                        MediaCodec mediaCodec4 = mediaCodec2;
                                                                        try {
                                                                            processOutputFrame(mediaCodec4, i4, mediaFormat, j5, j7, j3, orCreatePoseLandmarker, i5);
                                                                            mediaCodec2 = mediaCodec4;
                                                                            i3 = i5;
                                                                        } catch (Exception e8) {
                                                                            e = e8;
                                                                            mediaCodec2 = mediaCodec4;
                                                                            e.printStackTrace();
                                                                            LogUtil.e("error:" + e.getMessage());
                                                                            notifyComplete(i, false, 0, 4, e.getMessage());
                                                                            if (mediaCodec2 != null) {
                                                                            }
                                                                            if (mediaExtractor == null) {
                                                                            }
                                                                            mediaExtractor.release();
                                                                            return;
                                                                        } catch (Throwable th6) {
                                                                            th = th6;
                                                                            mediaCodec2 = mediaCodec4;
                                                                            if (mediaCodec2 != null) {
                                                                            }
                                                                            if (mediaExtractor == null) {
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                                mediaCodec2.releaseOutputBuffer(i4, z5);
                                                            } else {
                                                                mediaFormat = trackFormat;
                                                                i3 = i5;
                                                                z = true;
                                                                j2 = 1000;
                                                                try {
                                                                    Thread.sleep(5L);
                                                                } catch (InterruptedException unused8) {
                                                                }
                                                            }
                                                            z5 = z2;
                                                            j3 = j;
                                                            i5 = i3;
                                                            trackFormat = mediaFormat;
                                                            createDecoderByType = mediaCodec2;
                                                            z6 = z4;
                                                            z7 = z5;
                                                        }
                                                    } catch (Exception e9) {
                                                        e = e9;
                                                        mediaCodec2 = mediaCodec3;
                                                        e.printStackTrace();
                                                        LogUtil.e("error:" + e.getMessage());
                                                        notifyComplete(i, false, 0, 4, e.getMessage());
                                                        if (mediaCodec2 != null) {
                                                        }
                                                        if (mediaExtractor == null) {
                                                        }
                                                        mediaExtractor.release();
                                                        return;
                                                    } catch (Throwable th7) {
                                                        th = th7;
                                                        mediaCodec2 = mediaCodec3;
                                                        if (mediaCodec2 != null) {
                                                        }
                                                        if (mediaExtractor == null) {
                                                        }
                                                    }
                                                } else {
                                                    MediaCodec mediaCodec5 = createDecoderByType;
                                                    mediaCodec5.queueInputBuffer(dequeueInputBuffer, 0, readSampleData, mediaExtractor.getSampleTime(), 0);
                                                    mediaCodec2 = mediaCodec5;
                                                    mediaExtractor.advance();
                                                }
                                            }
                                            z4 = z6;
                                            dequeueOutputBuffer = mediaCodec2.dequeueOutputBuffer(bufferInfo, 1000L);
                                            if (dequeueOutputBuffer == -2) {
                                            }
                                        }
                                    }
                                    dequeueOutputBuffer = mediaCodec2.dequeueOutputBuffer(bufferInfo, 1000L);
                                    if (dequeueOutputBuffer == -2) {
                                    }
                                } catch (Exception e10) {
                                    e = e10;
                                }
                                z2 = z7;
                                if (z6) {
                                }
                                mediaCodec2 = createDecoderByType;
                                z3 = false;
                                z4 = z6;
                            } catch (Exception e11) {
                                e = e11;
                                mediaCodec2 = createDecoderByType;
                            }
                        }
                        mediaCodec2 = createDecoderByType;
                        i3 = i5;
                        z = true;
                        if (i3 == this.taskGeneration.get()) {
                            if (this.mTotalFrames != this.resultCache.size()) {
                                z = false;
                            }
                            this.isIntact = z;
                            notifyComplete(i3, true, this.resultCache.size(), 0, null);
                        }
                        if (mediaCodec2 != null) {
                            try {
                                mediaCodec2.stop();
                            } catch (Exception unused9) {
                            }
                            try {
                                mediaCodec2.release();
                            } catch (Exception unused10) {
                            }
                        }
                    } catch (Throwable th8) {
                        th = th8;
                        mediaCodec2 = createDecoderByType;
                    }
                } catch (Exception e12) {
                    e = e12;
                    mediaCodec2 = createDecoderByType;
                }
                mediaExtractor.release();
                return;
            }
            mediaCodec = null;
            try {
                notifyComplete(i, false, 0, 3, "视频元数据无效");
            } catch (Exception e13) {
                e = e13;
                mediaCodec2 = mediaCodec;
                e.printStackTrace();
                LogUtil.e("error:" + e.getMessage());
                notifyComplete(i, false, 0, 4, e.getMessage());
                if (mediaCodec2 != null) {
                }
                if (mediaExtractor == null) {
                }
                mediaExtractor.release();
                return;
            } catch (Throwable th9) {
                th = th9;
                mediaCodec2 = mediaCodec;
                if (mediaCodec2 != null) {
                }
                if (mediaExtractor == null) {
                }
            }
        }
        try {
            mediaExtractor.release();
        } catch (Exception unused11) {
        }
    }

    public boolean seekToDetectVideoAsync(long j) {
        synchronized (this.taskLock) {
            if (this.mCurrentProcessingUri == null) {
                LogUtil.e("PoseLandmarkerHelper4 seekToDetectVideoAsync: no video selected.");
                return false;
            }
            Long validTime = getValidTime(j);
            if (validTime != null && this.resultCache.containsKey(validTime) && isReadyAt(j)) {
                this.mCurrentPlayTime = validTime.longValue();
                LogUtil.d("PoseLandmarkerHelper4 seekToDetectVideoAsync: cache hit, playTime=" + j);
                return true;
            }
            LogUtil.d("PoseLandmarkerHelper4 seekToDetectVideoAsync: refill from " + j);
            cancelAsyncTask(false);
            trimCacheWindow(j);
            this.mHandler.removeCallbacks(this.mSeekRunnable);
            this.mPendingPlayTime = j;
            this.mHandler.postDelayed(this.mSeekRunnable, SEEK_DEBOUNCE_DELAY_MS);
            return false;
        }
    }

    public void setCacheWindow(long j, long j2, long j3) {
        this.mPastKeepMs = Math.max(0L, j);
        this.mFutureMaxMs = Math.max(this.mPlayReadyMs, j2);
        this.mPlayReadyMs = Math.max(0L, j3);
    }

    public void setSeekBackMs(long j) {
        this.mSeekBackMs = Math.max(0L, j);
    }
}
