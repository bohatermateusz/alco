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
import com.google.firebase.messaging.ServiceStarter;
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
import java.util.concurrent.atomic.AtomicBoolean;

/* loaded from: classes3.dex */
public class PoseLandmarkerHelper3 {
    private static final long SEEK_DEBOUNCE_DELAY_MS = 300;
    private Future<?> currentTask;
    private VideoFrameCallback mCallback;
    private final Context mContext;
    private Uri mCurrentProcessingUri;
    private PoseLandmarker mPoseLandmarker;
    private final ConcurrentSkipListMap<Long, PoseLandmarkerResult> resultCache = new ConcurrentSkipListMap<>();
    private long mInferenceIntervalMs = 100;
    private int mMaxCacheSize = ServiceStarter.ERROR_UNKNOWN;
    private long mInferenceBufferMs = ServiceStarter.ERROR_UNKNOWN * 100;
    private final ExecutorService singleExecutor = Executors.newSingleThreadExecutor();
    private boolean isIntact = false;
    private final AtomicBoolean isCancelled = new AtomicBoolean(false);
    private final Object taskLock = new Object();
    private long mLastPlayTime = 0;
    private long mVideoDurationMs = -1;
    private int mVideoWidth = 0;
    private int mVideoHeight = 0;
    private long mTotalFrames = -1;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private final Runnable mSeekRunnable = new Runnable() { // from class: com.hismith.hismithapp.pose.PoseLandmarkerHelper3.1
        @Override // java.lang.Runnable
        public void run() {
            PoseLandmarkerHelper3 poseLandmarkerHelper3 = PoseLandmarkerHelper3.this;
            poseLandmarkerHelper3.doSeekInternal(poseLandmarkerHelper3.mPendingPlayTime);
        }
    };
    private long mPendingPlayTime = -1;
    private long mCurrentPlayTime = 0;
    private final int STATE_BUFFERING = 2;
    private boolean isReadyCallback = false;
    private final int STATE_READY = 3;

    public interface VideoFrameCallback {
        void onCanPlayed();

        void onComplete(boolean z, int i, long j);

        void onFrameRatio(int i, int i2);
    }

    public PoseLandmarkerHelper3(Context context) {
        this.mContext = context;
    }

    private void cancelAsyncTask(boolean z) {
        Future<?> future = this.currentTask;
        if (future != null) {
            future.cancel(true);
            this.currentTask = null;
        }
        this.isCancelled.set(true);
        if (z) {
            this.resultCache.clear();
            this.mCurrentProcessingUri = null;
            this.mVideoDurationMs = -1L;
            this.mVideoWidth = 0;
            this.mVideoHeight = 0;
            this.mTotalFrames = -1L;
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
    
        if (r3 < r2.mLastPlayTime) goto L7;
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

    private boolean isSameUri(Uri uri, Uri uri2) {
        if (uri == null && uri2 == null) {
            return true;
        }
        if (uri == null || uri2 == null) {
            return false;
        }
        return uri.normalizeScheme().equals(uri2.normalizeScheme());
    }

    private void startDetectionTask(final long j) {
        this.isReadyCallback = false;
        this.isCancelled.set(false);
        this.currentTask = this.singleExecutor.submit(new Runnable() { // from class: com.hismith.hismithapp.pose.PoseLandmarkerHelper3$$ExternalSyntheticLambda0
            @Override // java.lang.Runnable
            public final void run() {
                PoseLandmarkerHelper3.this.m617xc03e2797(j);
            }
        });
    }

    private void trimCache(long j) {
        this.resultCache.headMap((ConcurrentSkipListMap<Long, PoseLandmarkerResult>) Long.valueOf(j), false).clear();
        this.resultCache.tailMap((ConcurrentSkipListMap<Long, PoseLandmarkerResult>) Long.valueOf((this.mMaxCacheSize * 100) + j), false).clear();
    }

    private Bitmap yuv420ToBitmap(ByteBuffer byteBuffer, int i, int i2, int i3) {
        int i4 = i * i2;
        int i5 = i4 / 2;
        int i6 = i4 + i5;
        if (byteBuffer.remaining() < i6) {
            LogUtil.e("Buffer size too small: " + byteBuffer.remaining() + " < " + i6);
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
            LogUtil.w("Unknown color format: " + i3 + ", trying NV12 fallback");
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
            LogUtil.e("YuvImage conversion error: " + e.getMessage());
            return null;
        }
    }

    public void cancelAll() {
        this.mHandler.removeCallbacks(this.mSeekRunnable);
        try {
            ExecutorService executorService = this.singleExecutor;
            if (executorService != null && !executorService.isShutdown()) {
                this.singleExecutor.shutdownNow();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        cancelAsyncTask(true);
    }

    public void detectVideoFileAsync(Uri uri, long j, VideoFrameCallback videoFrameCallback) {
        synchronized (this.taskLock) {
            this.mHandler.removeCallbacks(this.mSeekRunnable);
            if (isSameUri(this.mCurrentProcessingUri, uri)) {
                LogUtil.d("detectVideoFileAsync: same video, ignore.");
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
            startDetectionTask(0L);
        }
    }

    public PoseLandmarkerResult getCacheResult(long j) {
        Long validTime = getValidTime(j);
        if (validTime == null) {
            return null;
        }
        this.mCurrentPlayTime = validTime.longValue();
        PoseLandmarkerResult poseLandmarkerResult = getResultCache().get(validTime);
        LogUtil.printLongLog("PoseLandmarkerHelper3Result", poseLandmarkerResult != null ? poseLandmarkerResult.toString() : "null");
        return poseLandmarkerResult;
    }

    public ConcurrentSkipListMap<Long, PoseLandmarkerResult> getResultCache() {
        return this.resultCache;
    }

    public PoseLandmarkerResult getValidResult(long j) {
        Long floorKey = this.resultCache.floorKey(Long.valueOf(j));
        if (floorKey != null && j - floorKey.longValue() <= this.mInferenceIntervalMs) {
            return this.resultCache.get(floorKey);
        }
        return null;
    }

    public Long getValidTime(long j) {
        long j2 = this.mInferenceIntervalMs;
        Long valueOf = Long.valueOf((j / j2) * j2);
        Long floorKey = this.resultCache.floorKey(Long.valueOf(j));
        LogUtil.e("getValidTime playTme:" + j + ", key:" + floorKey);
        return (floorKey != null && j - floorKey.longValue() <= this.mInferenceIntervalMs) ? floorKey : valueOf;
    }

    /* JADX WARN: Removed duplicated region for block: B:181:0x0415 A[Catch: all -> 0x0427, TRY_LEAVE, TryCatch #4 {all -> 0x0427, blocks: (B:179:0x03f4, B:181:0x0415), top: B:178:0x03f4 }] */
    /* JADX WARN: Removed duplicated region for block: B:185:? A[RETURN, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:186:0x041d A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:197:0x0432 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:203:? A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:204:0x042a A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* renamed from: lambda$startDetectionTask$0$com-hismith-hismithapp-pose-PoseLandmarkerHelper3, reason: not valid java name */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    /* synthetic */ void m617xc03e2797(long j) {
        MediaExtractor mediaExtractor;
        VideoFrameCallback videoFrameCallback;
        MediaExtractor mediaExtractor2;
        MediaFormat trackFormat;
        long currentTimeMillis;
        long currentTimeMillis2;
        String str;
        PoseLandmarker poseLandmarker;
        MediaCodec.BufferInfo bufferInfo;
        MediaExtractor mediaExtractor3;
        MediaFormat mediaFormat;
        long j2;
        long j3;
        ByteBuffer outputBuffer;
        MPImage mPImage;
        VideoFrameCallback videoFrameCallback2;
        String str2 = "color-format";
        PoseLandmarker orCreatePoseLandmarker = getOrCreatePoseLandmarker(j);
        boolean z = false;
        if (orCreatePoseLandmarker == null) {
            VideoFrameCallback videoFrameCallback3 = this.mCallback;
            if (videoFrameCallback3 != null) {
                videoFrameCallback3.onComplete(false, 0, 0L);
                return;
            }
            return;
        }
        MediaCodec mediaCodec = null;
        try {
            mediaExtractor2 = new MediaExtractor();
            try {
                mediaExtractor2.setDataSource(this.mContext, this.mCurrentProcessingUri, (Map<String, String>) null);
                int i = 0;
                while (true) {
                    if (i >= mediaExtractor2.getTrackCount()) {
                        i = -1;
                        break;
                    }
                    try {
                        String string = mediaExtractor2.getTrackFormat(i).getString("mime");
                        if (string != null && string.startsWith("video/")) {
                            break;
                        } else {
                            i++;
                        }
                    } catch (Exception e) {
                        e = e;
                        mediaExtractor = mediaExtractor2;
                        try {
                            e.printStackTrace();
                            LogUtil.e("Exception:" + e.getMessage());
                            videoFrameCallback = this.mCallback;
                            if (videoFrameCallback != null) {
                            }
                            if (mediaCodec != null) {
                            }
                            if (mediaExtractor == null) {
                            }
                            try {
                                mediaExtractor.release();
                            } catch (Exception unused) {
                                return;
                            }
                        } catch (Throwable th) {
                            th = th;
                            if (mediaCodec != null) {
                                try {
                                    mediaCodec.stop();
                                } catch (Exception unused2) {
                                }
                                try {
                                    mediaCodec.release();
                                } catch (Exception unused3) {
                                }
                            }
                            if (mediaExtractor == null) {
                                throw th;
                            }
                            try {
                                mediaExtractor.release();
                                throw th;
                            } catch (Exception unused4) {
                                throw th;
                            }
                        }
                    } catch (Throwable th2) {
                        th = th2;
                        mediaExtractor = mediaExtractor2;
                        if (mediaCodec != null) {
                        }
                        if (mediaExtractor == null) {
                        }
                    }
                }
                mediaExtractor2.selectTrack(i);
                trackFormat = mediaExtractor2.getTrackFormat(i);
                this.mVideoWidth = trackFormat.getInteger(ViewHierarchyConstants.DIMENSION_WIDTH_KEY);
                this.mVideoHeight = trackFormat.getInteger(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY);
                long j4 = (trackFormat.containsKey("durationUs") ? trackFormat.getLong("durationUs") : -1L) / 1000;
                this.mVideoDurationMs = j4;
                this.mTotalFrames = j4 / this.mInferenceIntervalMs;
                VideoFrameCallback videoFrameCallback4 = this.mCallback;
                if (videoFrameCallback4 != null) {
                    videoFrameCallback4.onFrameRatio(this.mVideoWidth, this.mVideoHeight);
                    LogUtil.e("onFrameRatio time :" + System.currentTimeMillis());
                }
            } catch (Exception e2) {
                e = e2;
                mediaExtractor = mediaExtractor2;
            } catch (Throwable th3) {
                th = th3;
                mediaExtractor = mediaExtractor2;
            }
        } catch (Exception e3) {
            e = e3;
            mediaCodec = null;
            mediaExtractor = null;
        } catch (Throwable th4) {
            th = th4;
            mediaCodec = null;
            mediaExtractor = null;
        }
        if (this.mVideoDurationMs > 0 && this.mVideoWidth != 0 && this.mVideoHeight != 0) {
            MediaCodec createDecoderByType = MediaCodec.createDecoderByType(trackFormat.getString("mime"));
            try {
                createDecoderByType.configure(trackFormat, (Surface) null, (MediaCrypto) null, 0);
                createDecoderByType.start();
                LogUtil.e("MediaCodec start time :" + System.currentTimeMillis());
                long max = Math.max((j - 5000) * 1000, 0L);
                LogUtil.e("跳转推理：" + j + ", 实际跳转：" + max);
                mediaExtractor2.seekTo(max, 2);
                MediaCodec.BufferInfo bufferInfo2 = new MediaCodec.BufferInfo();
                boolean z2 = false;
                boolean z3 = false;
                while (!z2) {
                    if (this.isCancelled.get()) {
                        break;
                    }
                    if (this.resultCache.isEmpty()) {
                        mediaExtractor = mediaExtractor2;
                    } else {
                        mediaExtractor = mediaExtractor2;
                        try {
                            try {
                                if (this.resultCache.tailMap((ConcurrentSkipListMap<Long, PoseLandmarkerResult>) Long.valueOf(this.mCurrentPlayTime)).size() > this.mMaxCacheSize) {
                                    long j5 = this.mCurrentPlayTime;
                                    long j6 = this.mInferenceIntervalMs;
                                    if (j5 > j6 * 10) {
                                        this.resultCache.headMap((ConcurrentSkipListMap<Long, PoseLandmarkerResult>) Long.valueOf(j5 - (j6 * 10)), z).clear();
                                    }
                                    try {
                                        LogUtil.e("休眠");
                                        Thread.sleep(50L);
                                        mediaExtractor2 = mediaExtractor;
                                    } catch (InterruptedException unused5) {
                                    }
                                }
                            } catch (Throwable th5) {
                                th = th5;
                                mediaCodec = createDecoderByType;
                                if (mediaCodec != null) {
                                }
                                if (mediaExtractor == null) {
                                }
                            }
                        } catch (Exception e4) {
                            e = e4;
                            mediaCodec = createDecoderByType;
                            e.printStackTrace();
                            LogUtil.e("Exception:" + e.getMessage());
                            videoFrameCallback = this.mCallback;
                            if (videoFrameCallback != null) {
                                videoFrameCallback.onComplete(false, 0, 0L);
                            }
                            if (mediaCodec != null) {
                                try {
                                    mediaCodec.stop();
                                } catch (Exception unused6) {
                                }
                                try {
                                    mediaCodec.release();
                                } catch (Exception unused7) {
                                }
                            }
                            if (mediaExtractor == null) {
                                return;
                            }
                            mediaExtractor.release();
                        }
                    }
                    if (z3) {
                        mediaExtractor2 = mediaExtractor;
                        currentTimeMillis = 0;
                        currentTimeMillis2 = 0;
                    } else {
                        try {
                            currentTimeMillis = System.currentTimeMillis();
                            int dequeueInputBuffer = createDecoderByType.dequeueInputBuffer(Renderer.DEFAULT_DURATION_TO_PROGRESS_US);
                            if (dequeueInputBuffer >= 0) {
                                mediaExtractor2 = mediaExtractor;
                                try {
                                    int readSampleData = mediaExtractor2.readSampleData(createDecoderByType.getInputBuffer(dequeueInputBuffer), z ? 1 : 0);
                                    if (readSampleData < 0) {
                                        MediaCodec mediaCodec2 = createDecoderByType;
                                        try {
                                            mediaCodec2.queueInputBuffer(dequeueInputBuffer, 0, 0, 0L, 4);
                                            createDecoderByType = mediaCodec2;
                                            z3 = true;
                                        } catch (Exception e5) {
                                            e = e5;
                                            mediaExtractor = mediaExtractor2;
                                            mediaCodec = mediaCodec2;
                                            e.printStackTrace();
                                            LogUtil.e("Exception:" + e.getMessage());
                                            videoFrameCallback = this.mCallback;
                                            if (videoFrameCallback != null) {
                                            }
                                            if (mediaCodec != null) {
                                            }
                                            if (mediaExtractor == null) {
                                            }
                                            mediaExtractor.release();
                                        } catch (Throwable th6) {
                                            th = th6;
                                            mediaExtractor = mediaExtractor2;
                                            mediaCodec = mediaCodec2;
                                            if (mediaCodec != null) {
                                            }
                                            if (mediaExtractor == null) {
                                            }
                                        }
                                    } else {
                                        MediaCodec mediaCodec3 = createDecoderByType;
                                        try {
                                            mediaCodec3.queueInputBuffer(dequeueInputBuffer, 0, readSampleData, mediaExtractor2.getSampleTime(), 0);
                                            createDecoderByType = mediaCodec3;
                                            mediaExtractor2.advance();
                                        } catch (Exception e6) {
                                            e = e6;
                                            createDecoderByType = mediaCodec3;
                                            mediaCodec = createDecoderByType;
                                            mediaExtractor = mediaExtractor2;
                                            e.printStackTrace();
                                            LogUtil.e("Exception:" + e.getMessage());
                                            videoFrameCallback = this.mCallback;
                                            if (videoFrameCallback != null) {
                                            }
                                            if (mediaCodec != null) {
                                            }
                                            if (mediaExtractor == null) {
                                            }
                                            mediaExtractor.release();
                                        } catch (Throwable th7) {
                                            th = th7;
                                            createDecoderByType = mediaCodec3;
                                            mediaCodec = createDecoderByType;
                                            mediaExtractor = mediaExtractor2;
                                            if (mediaCodec != null) {
                                            }
                                            if (mediaExtractor == null) {
                                            }
                                        }
                                    }
                                } catch (Exception e7) {
                                    e = e7;
                                } catch (Throwable th8) {
                                    th = th8;
                                }
                            } else {
                                mediaExtractor2 = mediaExtractor;
                            }
                            currentTimeMillis2 = System.currentTimeMillis();
                        } catch (Exception e8) {
                            e = e8;
                            mediaCodec = createDecoderByType;
                            e.printStackTrace();
                            LogUtil.e("Exception:" + e.getMessage());
                            videoFrameCallback = this.mCallback;
                            if (videoFrameCallback != null) {
                            }
                            if (mediaCodec != null) {
                            }
                            if (mediaExtractor == null) {
                            }
                            mediaExtractor.release();
                        } catch (Throwable th9) {
                            th = th9;
                            mediaCodec = createDecoderByType;
                            if (mediaCodec != null) {
                            }
                            if (mediaExtractor == null) {
                            }
                        }
                    }
                    long currentTimeMillis3 = System.currentTimeMillis();
                    int dequeueOutputBuffer = createDecoderByType.dequeueOutputBuffer(bufferInfo2, Renderer.DEFAULT_DURATION_TO_PROGRESS_US);
                    if (dequeueOutputBuffer < 0) {
                        str = str2;
                        poseLandmarker = orCreatePoseLandmarker;
                        bufferInfo = bufferInfo2;
                        mediaExtractor3 = mediaExtractor2;
                        mediaFormat = trackFormat;
                    } else if ((bufferInfo2.flags & 4) != 0) {
                        str = str2;
                        poseLandmarker = orCreatePoseLandmarker;
                        bufferInfo = bufferInfo2;
                        mediaExtractor3 = mediaExtractor2;
                        mediaFormat = trackFormat;
                        z2 = true;
                    } else {
                        long j7 = bufferInfo2.presentationTimeUs / 1000;
                        LogUtil.e("输出frameTimeMs" + j7);
                        MediaCodec.BufferInfo bufferInfo3 = bufferInfo2;
                        boolean z4 = z2;
                        long j8 = this.mInferenceIntervalMs;
                        long j9 = j8 * (j7 / j8);
                        if (j7 >= j) {
                            bufferInfo = bufferInfo3;
                            mediaExtractor3 = mediaExtractor2;
                            if (this.resultCache.containsKey(Long.valueOf(j9)) || (outputBuffer = createDecoderByType.getOutputBuffer(dequeueOutputBuffer)) == null) {
                                str = str2;
                                poseLandmarker = orCreatePoseLandmarker;
                            } else {
                                str = str2;
                                mediaFormat = trackFormat;
                                Bitmap yuv420ToBitmap = yuv420ToBitmap(outputBuffer, this.mVideoWidth, this.mVideoHeight, trackFormat.containsKey(str2) ? trackFormat.getInteger(str2) : 21);
                                if (yuv420ToBitmap != null) {
                                    try {
                                        mPImage = new BitmapImageBuilder(yuv420ToBitmap).build();
                                        try {
                                            j3 = System.currentTimeMillis();
                                            PoseLandmarkerResult detectForVideo = orCreatePoseLandmarker.detectForVideo(mPImage, j7);
                                            j2 = System.currentTimeMillis();
                                            if (detectForVideo == null || this.isCancelled.get()) {
                                                poseLandmarker = orCreatePoseLandmarker;
                                            } else {
                                                poseLandmarker = orCreatePoseLandmarker;
                                                this.resultCache.put(Long.valueOf(j9), detectForVideo);
                                                this.mLastPlayTime = j7;
                                                if (this.resultCache.size() > 100 && (videoFrameCallback2 = this.mCallback) != null && !this.isReadyCallback) {
                                                    videoFrameCallback2.onCanPlayed();
                                                    this.isReadyCallback = true;
                                                }
                                                LogUtil.w("resultCache size:" + this.resultCache.size() + ", timestampMs:" + j7 + ", key:" + j9 + ", mVideoDurationMs:" + this.mVideoDurationMs);
                                            }
                                            if (mPImage != null) {
                                                try {
                                                    mPImage.close();
                                                } catch (Exception unused8) {
                                                }
                                            }
                                            if (yuv420ToBitmap != null && !yuv420ToBitmap.isRecycled()) {
                                                yuv420ToBitmap.recycle();
                                            }
                                            createDecoderByType.releaseOutputBuffer(dequeueOutputBuffer, true);
                                            z2 = z4;
                                            long j10 = currentTimeMillis2 - currentTimeMillis;
                                            long currentTimeMillis4 = System.currentTimeMillis() - currentTimeMillis3;
                                            long j11 = j2 - j3;
                                            LogUtil.e("FrameTime, inputTime :" + j10 + " ,output :" + currentTimeMillis4 + " ,DetectTime:" + j11 + " ,totaltime:" + (j10 + currentTimeMillis4 + j11));
                                            bufferInfo2 = bufferInfo;
                                            str2 = str;
                                            mediaExtractor2 = mediaExtractor3;
                                            trackFormat = mediaFormat;
                                            orCreatePoseLandmarker = poseLandmarker;
                                            z = false;
                                        } catch (Throwable th10) {
                                            th = th10;
                                            if (mPImage != null) {
                                                try {
                                                    mPImage.close();
                                                } catch (Exception unused9) {
                                                }
                                            }
                                            if (yuv420ToBitmap == null) {
                                                throw th;
                                            }
                                            if (yuv420ToBitmap.isRecycled()) {
                                                throw th;
                                            }
                                            yuv420ToBitmap.recycle();
                                            throw th;
                                        }
                                    } catch (Throwable th11) {
                                        th = th11;
                                        mPImage = null;
                                    }
                                } else {
                                    poseLandmarker = orCreatePoseLandmarker;
                                    j3 = 0;
                                    j2 = 0;
                                    createDecoderByType.releaseOutputBuffer(dequeueOutputBuffer, true);
                                    z2 = z4;
                                    long j102 = currentTimeMillis2 - currentTimeMillis;
                                    long currentTimeMillis42 = System.currentTimeMillis() - currentTimeMillis3;
                                    long j112 = j2 - j3;
                                    LogUtil.e("FrameTime, inputTime :" + j102 + " ,output :" + currentTimeMillis42 + " ,DetectTime:" + j112 + " ,totaltime:" + (j102 + currentTimeMillis42 + j112));
                                    bufferInfo2 = bufferInfo;
                                    str2 = str;
                                    mediaExtractor2 = mediaExtractor3;
                                    trackFormat = mediaFormat;
                                    orCreatePoseLandmarker = poseLandmarker;
                                    z = false;
                                }
                            }
                        } else {
                            str = str2;
                            poseLandmarker = orCreatePoseLandmarker;
                            bufferInfo = bufferInfo3;
                            mediaExtractor3 = mediaExtractor2;
                        }
                        mediaFormat = trackFormat;
                        j3 = 0;
                        j2 = 0;
                        createDecoderByType.releaseOutputBuffer(dequeueOutputBuffer, true);
                        z2 = z4;
                        long j1022 = currentTimeMillis2 - currentTimeMillis;
                        long currentTimeMillis422 = System.currentTimeMillis() - currentTimeMillis3;
                        long j1122 = j2 - j3;
                        LogUtil.e("FrameTime, inputTime :" + j1022 + " ,output :" + currentTimeMillis422 + " ,DetectTime:" + j1122 + " ,totaltime:" + (j1022 + currentTimeMillis422 + j1122));
                        bufferInfo2 = bufferInfo;
                        str2 = str;
                        mediaExtractor2 = mediaExtractor3;
                        trackFormat = mediaFormat;
                        orCreatePoseLandmarker = poseLandmarker;
                        z = false;
                    }
                    j3 = 0;
                    j2 = 0;
                    long j10222 = currentTimeMillis2 - currentTimeMillis;
                    long currentTimeMillis4222 = System.currentTimeMillis() - currentTimeMillis3;
                    long j11222 = j2 - j3;
                    LogUtil.e("FrameTime, inputTime :" + j10222 + " ,output :" + currentTimeMillis4222 + " ,DetectTime:" + j11222 + " ,totaltime:" + (j10222 + currentTimeMillis4222 + j11222));
                    bufferInfo2 = bufferInfo;
                    str2 = str;
                    mediaExtractor2 = mediaExtractor3;
                    trackFormat = mediaFormat;
                    orCreatePoseLandmarker = poseLandmarker;
                    z = false;
                }
                mediaExtractor = mediaExtractor2;
                this.isIntact = this.mTotalFrames == ((long) this.resultCache.size());
                if (this.mCallback != null) {
                    boolean z5 = !this.isCancelled.get();
                    if (!this.isReadyCallback) {
                        this.mCallback.onCanPlayed();
                        this.isReadyCallback = true;
                    }
                    this.mCallback.onComplete(z5, this.resultCache.size(), 0L);
                }
                if (createDecoderByType != null) {
                    try {
                        createDecoderByType.stop();
                    } catch (Exception unused10) {
                    }
                    try {
                        createDecoderByType.release();
                    } catch (Exception unused11) {
                    }
                }
            } catch (Exception e9) {
                e = e9;
                mediaExtractor = mediaExtractor2;
            } catch (Throwable th12) {
                th = th12;
                mediaExtractor = mediaExtractor2;
            }
            mediaExtractor.release();
        }
        mediaExtractor = mediaExtractor2;
        try {
            VideoFrameCallback videoFrameCallback5 = this.mCallback;
            if (videoFrameCallback5 != null) {
                videoFrameCallback5.onComplete(false, 0, 0L);
            }
            try {
                mediaExtractor.release();
            } catch (Exception unused12) {
            }
        } catch (Exception e10) {
            e = e10;
            mediaCodec = null;
            e.printStackTrace();
            LogUtil.e("Exception:" + e.getMessage());
            videoFrameCallback = this.mCallback;
            if (videoFrameCallback != null) {
            }
            if (mediaCodec != null) {
            }
            if (mediaExtractor == null) {
            }
            mediaExtractor.release();
        } catch (Throwable th13) {
            th = th13;
            mediaCodec = null;
            if (mediaCodec != null) {
            }
            if (mediaExtractor == null) {
            }
        }
    }

    public boolean seekToDetectVideoAsync(long j) {
        synchronized (this.taskLock) {
            if (this.mCurrentProcessingUri == null) {
                LogUtil.e("seekToDetectVideoAsync: no video selected.");
                return false;
            }
            Long validTime = getValidTime(j);
            if (validTime != null && this.resultCache.containsKey(validTime)) {
                LogUtil.e("seekToDetectVideoAsync: 跳转位置已在缓存中");
                return true;
            }
            this.mCurrentPlayTime = validTime.longValue();
            cancelAsyncTask(false);
            trimCache(validTime.longValue());
            this.mHandler.removeCallbacks(this.mSeekRunnable);
            this.mPendingPlayTime = j;
            this.mHandler.postDelayed(this.mSeekRunnable, SEEK_DEBOUNCE_DELAY_MS);
            return false;
        }
    }

    public void setMaxCacheSize(int i) {
        this.mMaxCacheSize = i;
    }
}
