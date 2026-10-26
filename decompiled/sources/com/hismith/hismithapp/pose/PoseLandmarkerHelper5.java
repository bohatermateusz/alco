package com.hismith.hismithapp.pose;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.MediaCodec;
import android.media.MediaCrypto;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.view.Surface;
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
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/* loaded from: classes3.dex */
public class PoseLandmarkerHelper5 {
    private static final long DEQUEUE_TIMEOUT_US = 1000;
    public static final int ERROR_DECODE_EXCEPTION = 4;
    public static final int ERROR_INVALID_VIDEO_META = 3;
    public static final int ERROR_MODEL_INIT_FAILED = 1;
    public static final int ERROR_NONE = 0;
    public static final int ERROR_NO_VIDEO_TRACK = 2;
    public static final int ERROR_UNSUPPORTED_COLOR_FORMAT = 5;
    private static final long SEEK_DEBOUNCE_DELAY_MS = 300;
    private static final String TAG = "PoseLandmarkerHelper5";
    private static final int TASK_ERROR_THRESHOLD = 20;
    private Future<?> currentTask;
    private VideoFrameCallback mCallback;
    private final Context mContext;
    private Uri mCurrentProcessingUri;
    private PoseLandmarker mPoseLandmarker;
    private final ConcurrentSkipListMap<Long, PoseLandmarkerResult> resultCache = new ConcurrentSkipListMap<>();
    private long mInferenceIntervalMs = 100;
    private long mPastKeepMs = 20000;
    private long mFutureMaxMs = 50000;
    private long mPlayReadyMs = 5000;
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
    private int mStorageWidth = 0;
    private int mStorageHeight = 0;
    private int mVideoRotation = 0;
    private long mTotalFrames = -1;
    private int mTaskCachedFrames = 0;
    private int mTaskColorSkipped = 0;
    private int mTaskDetectErrors = 0;
    private volatile long mTaskLastCached = -1;
    private long statFrames = 0;
    private long statYuvNs = 0;
    private long statRotateNs = 0;
    private long statMpImageNs = 0;
    private long statDetectNs = 0;
    private long statFrameNs = 0;
    private long statYuvMaxNs = 0;
    private long statRotateMaxNs = 0;
    private long statMpImageMaxNs = 0;
    private long statDetectMaxNs = 0;
    private long statFrameMaxNs = 0;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private final Runnable mSeekRunnable = new Runnable() { // from class: com.hismith.hismithapp.pose.PoseLandmarkerHelper5.1
        @Override // java.lang.Runnable
        public void run() {
            PoseLandmarkerHelper5 poseLandmarkerHelper5 = PoseLandmarkerHelper5.this;
            poseLandmarkerHelper5.doSeekInternal(poseLandmarkerHelper5.mPendingPlayTime);
        }
    };
    private long mPendingPlayTime = -1;

    public interface VideoFrameCallback {
        void onCacheProgress(int i, long j);

        void onCanPlayed();

        void onComplete(boolean z, int i, long j, int i2, String str);

        void onFrameRatio(int i, int i2);
    }

    public PoseLandmarkerHelper5(Context context) {
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
            this.mStorageWidth = 0;
            this.mStorageHeight = 0;
            this.mVideoRotation = 0;
            this.mTotalFrames = -1L;
            this.mCurrentPlayTime = 0L;
            this.mTaskLastCached = -1L;
            this.mTaskCachedFrames = 0;
            this.mTaskColorSkipped = 0;
            this.mTaskDetectErrors = 0;
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

    /* JADX WARN: Code restructure failed: missing block: B:21:0x000b, code lost:
    
        if (r3 <= r2.mLastPlayTime) goto L7;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private synchronized PoseLandmarker getOrCreatePoseLandmarker(long j) {
        if (j != 0) {
        }
        PoseLandmarker poseLandmarker = this.mPoseLandmarker;
        if (poseLandmarker != null) {
            try {
                poseLandmarker.close();
            } catch (Exception unused) {
            }
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
        if (this.isIntact) {
            return true;
        }
        Long lastKey = this.resultCache.lastKey();
        if (lastKey == null) {
            return false;
        }
        if (lastKey.longValue() - j >= this.mPlayReadyMs) {
            return true;
        }
        return this.mVideoDurationMs > 0 && lastKey.longValue() >= this.mVideoDurationMs - Math.max(100L, this.mInferenceIntervalMs * 2);
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

    private void notifyComplete(int i, boolean z, int i2, int i3, String str, boolean z2, boolean z3) {
        if (i != this.taskGeneration.get() || this.mCallback == null) {
            return;
        }
        if (this.statFrames > 0) {
            LogUtil.d(TAG + String.format(Locale.US, " 每帧耗时统计: 帧数=%d | yuv→Bitmap 平均=%.2fms 最大=%.2f | 旋转 平均=%.2f 最大=%.2f | MPImage 平均=%.2f 最大=%.2f | detectForVideo 平均=%.2f 最大=%.2f | 整帧 平均=%.2f 最大=%.2f", Long.valueOf(this.statFrames), Float.valueOf(nsToMs(this.statYuvNs)), Float.valueOf(nsToMs(this.statYuvMaxNs)), Float.valueOf(nsToMs(this.statRotateNs)), Float.valueOf(nsToMs(this.statRotateMaxNs)), Float.valueOf(nsToMs(this.statMpImageNs)), Float.valueOf(nsToMs(this.statMpImageMaxNs)), Float.valueOf(nsToMs(this.statDetectNs)), Float.valueOf(nsToMs(this.statDetectMaxNs)), Float.valueOf(nsToMs(this.statFrameNs)), Float.valueOf(nsToMs(this.statFrameMaxNs))));
        }
        if (z3) {
            return;
        }
        if (z2 && !this.isReadyCallback) {
            this.mCallback.onCanPlayed();
            this.isReadyCallback = true;
        }
        LogUtil.d("PoseLandmarkerHelper5 onComplete success=" + z + ", frames=" + i2 + ", errorCode=" + i3 + ", errorMsg=" + str);
        this.mCallback.onComplete(z, i2, 0L, i3, str);
    }

    private static float nsToMs(long j) {
        return j / 1000000.0f;
    }

    /* JADX WARN: Can't wrap try/catch for region: R(11:(2:18|(16:21|(1:23)|24|25|26|27|28|29|30|31|32|(3:34|35|(9:37|39|40|(4:42|43|44|45)(1:76)|46|(1:52)|(2:60|61)|(1:56)|59))(1:80)|79|(0)|(0)|59))(1:107)|28|29|30|31|32|(0)(0)|79|(0)|(0)|59) */
    /* JADX WARN: Can't wrap try/catch for region: R(17:16|(2:18|(16:21|(1:23)|24|25|26|27|28|29|30|31|32|(3:34|35|(9:37|39|40|(4:42|43|44|45)(1:76)|46|(1:52)|(2:60|61)|(1:56)|59))(1:80)|79|(0)|(0)|59))(1:107)|106|24|25|26|27|28|29|30|31|32|(0)(0)|79|(0)|(0)|59) */
    /* JADX WARN: Code restructure failed: missing block: B:102:0x0211, code lost:
    
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:103:0x0212, code lost:
    
        r21 = r15;
     */
    /* JADX WARN: Code restructure failed: missing block: B:104:0x020d, code lost:
    
        r0 = th;
     */
    /* JADX WARN: Code restructure failed: missing block: B:105:0x020e, code lost:
    
        r21 = r15;
     */
    /* JADX WARN: Code restructure failed: missing block: B:57:0x01fd, code lost:
    
        if (r21.isRecycled() == false) goto L63;
     */
    /* JADX WARN: Code restructure failed: missing block: B:58:0x01ff, code lost:
    
        r21.recycle();
     */
    /* JADX WARN: Code restructure failed: missing block: B:69:0x022a, code lost:
    
        if (r21.isRecycled() == false) goto L63;
     */
    /* JADX WARN: Code restructure failed: missing block: B:82:0x0207, code lost:
    
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:83:0x0208, code lost:
    
        r16 = r15;
     */
    /* JADX WARN: Code restructure failed: missing block: B:84:0x0214, code lost:
    
        r4 = -1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:85:0x0203, code lost:
    
        r0 = th;
     */
    /* JADX WARN: Code restructure failed: missing block: B:86:0x0204, code lost:
    
        r16 = r15;
     */
    /* JADX WARN: Code restructure failed: missing block: B:87:0x022f, code lost:
    
        if (r16 != null) goto L102;
     */
    /* JADX WARN: Code restructure failed: missing block: B:88:0x0234, code lost:
    
        if (r21 != null) goto L87;
     */
    /* JADX WARN: Code restructure failed: missing block: B:90:0x023a, code lost:
    
        if (r21.isRecycled() == false) goto L89;
     */
    /* JADX WARN: Code restructure failed: missing block: B:91:0x023c, code lost:
    
        r21.recycle();
     */
    /* JADX WARN: Code restructure failed: missing block: B:92:0x023f, code lost:
    
        throw r0;
     */
    /* JADX WARN: Code restructure failed: missing block: B:93:?, code lost:
    
        throw r0;
     */
    /* JADX WARN: Code restructure failed: missing block: B:94:?, code lost:
    
        throw r0;
     */
    /* JADX WARN: Code restructure failed: missing block: B:96:0x0231, code lost:
    
        r16.close();
     */
    /* JADX WARN: Code restructure failed: missing block: B:99:0x020b, code lost:
    
        r0 = e;
     */
    /* JADX WARN: Removed duplicated region for block: B:34:0x00bc  */
    /* JADX WARN: Removed duplicated region for block: B:56:0x01f9  */
    /* JADX WARN: Removed duplicated region for block: B:60:0x01f4 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:68:0x0226  */
    /* JADX WARN: Removed duplicated region for block: B:70:0x0221 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:80:0x01ee  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private long processOutputFrame(MediaCodec mediaCodec, int i, MediaFormat mediaFormat, long j, long j2, long j3, PoseLandmarker poseLandmarker, int i2) {
        long j4;
        Bitmap bitmap;
        MPImage mPImage;
        Bitmap bitmap2;
        long j5;
        PoseLandmarkerResult detectForVideo;
        ByteBuffer outputBuffer = mediaCodec.getOutputBuffer(i);
        if (outputBuffer == null) {
            return -1L;
        }
        int integer = mediaFormat.containsKey("color-format") ? mediaFormat.getInteger("color-format") : 21;
        if (integer != 19 && integer != 21) {
            this.mTaskColorSkipped++;
            LogUtil.e("PoseLandmarkerHelper5 不支持的色彩格式 colorFormat=" + integer + "，跳过该帧推理");
            return -1L;
        }
        long nanoTime = System.nanoTime();
        long nanoTime2 = System.nanoTime();
        Bitmap yuv420ToBitmap = yuv420ToBitmap(outputBuffer, this.mStorageWidth, this.mStorageHeight, integer);
        if (yuv420ToBitmap == null) {
            return -1L;
        }
        long nanoTime3 = System.nanoTime() - nanoTime2;
        try {
            if (this.mVideoRotation != 0) {
                long nanoTime4 = System.nanoTime();
                bitmap = rotateBitmap(yuv420ToBitmap, this.mVideoRotation);
                j4 = System.nanoTime() - nanoTime4;
                if (bitmap != null && bitmap != yuv420ToBitmap) {
                    if (!yuv420ToBitmap.isRecycled()) {
                        yuv420ToBitmap.recycle();
                    }
                    long j6 = j4;
                    mPImage = null;
                    long nanoTime5 = System.nanoTime();
                    bitmap2 = bitmap;
                    MPImage build = new BitmapImageBuilder(bitmap).build();
                    long nanoTime6 = System.nanoTime() - nanoTime5;
                    long nanoTime7 = System.nanoTime();
                    detectForVideo = poseLandmarker.detectForVideo(build, j);
                    long nanoTime8 = System.nanoTime() - nanoTime7;
                    if (detectForVideo == null) {
                        mPImage = build;
                        if (i2 == this.taskGeneration.get()) {
                            this.statFrames++;
                            this.statYuvNs += nanoTime3;
                            this.statYuvMaxNs = Math.max(this.statYuvMaxNs, nanoTime3);
                            this.statRotateNs += j6;
                            this.statRotateMaxNs = Math.max(this.statRotateMaxNs, j6);
                            this.statMpImageNs += nanoTime6;
                            this.statMpImageMaxNs = Math.max(this.statMpImageMaxNs, nanoTime6);
                            this.statDetectNs += nanoTime8;
                            this.statDetectMaxNs = Math.max(this.statDetectMaxNs, nanoTime8);
                            long nanoTime9 = System.nanoTime() - nanoTime;
                            this.statFrameNs += nanoTime9;
                            this.statFrameMaxNs = Math.max(this.statFrameMaxNs, nanoTime9);
                            LogUtil.d(TAG + String.format(Locale.US, " 帧耗时 key=%d | yuv=%.2fms 旋转=%.2f MPImage=%.2f detect=%.2f 整帧=%.2f", Long.valueOf(j2), Float.valueOf(nsToMs(nanoTime3)), Float.valueOf(nsToMs(j6)), Float.valueOf(nsToMs(nanoTime6)), Float.valueOf(nsToMs(nanoTime8)), Float.valueOf(nsToMs(nanoTime9))));
                            this.resultCache.put(Long.valueOf(j2), detectForVideo);
                            this.mTaskCachedFrames++;
                            try {
                                this.mLastPlayTime = j;
                                VideoFrameCallback videoFrameCallback = this.mCallback;
                                if (videoFrameCallback != null) {
                                    j5 = j2;
                                    try {
                                        videoFrameCallback.onCacheProgress(this.resultCache.size(), j5);
                                    } catch (Exception e) {
                                        e = e;
                                        e.printStackTrace();
                                        this.mTaskDetectErrors++;
                                        if (mPImage != null) {
                                            try {
                                                mPImage.close();
                                            } catch (Exception unused) {
                                            }
                                        }
                                        if (bitmap2 != null) {
                                        }
                                        return j5;
                                    }
                                } else {
                                    j5 = j2;
                                }
                                LogUtil.w("PoseLandmarkerHelper5 已缓存 size=" + this.resultCache.size() + ", key=" + j5 + ", duration=" + this.mVideoDurationMs);
                                if (this.mCallback != null && !this.isReadyCallback && isReadyAt(j3)) {
                                    this.mCallback.onCanPlayed();
                                    this.isReadyCallback = true;
                                    LogUtil.d("PoseLandmarkerHelper5 onCanPlayed 触发，目标=" + j3 + ", 已缓存到=" + j5);
                                }
                                if (mPImage != null) {
                                    try {
                                        mPImage.close();
                                    } catch (Exception unused2) {
                                    }
                                }
                                if (bitmap2 != null) {
                                }
                            } catch (Exception e2) {
                                e = e2;
                                j5 = j2;
                            }
                            return j5;
                        }
                    } else {
                        mPImage = build;
                    }
                    j5 = -1;
                    if (mPImage != null) {
                    }
                    if (bitmap2 != null) {
                    }
                    return j5;
                }
            } else {
                j4 = 0;
            }
            MPImage build2 = new BitmapImageBuilder(bitmap).build();
            long nanoTime62 = System.nanoTime() - nanoTime5;
            long nanoTime72 = System.nanoTime();
            detectForVideo = poseLandmarker.detectForVideo(build2, j);
            long nanoTime82 = System.nanoTime() - nanoTime72;
            if (detectForVideo == null) {
            }
            j5 = -1;
            if (mPImage != null) {
            }
            if (bitmap2 != null) {
            }
            return j5;
        } catch (Throwable th) {
            th = th;
        }
        bitmap = yuv420ToBitmap;
        long j62 = j4;
        mPImage = null;
        long nanoTime52 = System.nanoTime();
        bitmap2 = bitmap;
    }

    private void resetFrameStats() {
        this.statFrames = 0L;
        this.statDetectNs = 0L;
        this.statMpImageNs = 0L;
        this.statRotateNs = 0L;
        this.statYuvNs = 0L;
        this.statFrameNs = 0L;
        this.statDetectMaxNs = 0L;
        this.statMpImageMaxNs = 0L;
        this.statRotateMaxNs = 0L;
        this.statYuvMaxNs = 0L;
        this.statFrameMaxNs = 0L;
    }

    private int resolveVideoRotation(MediaFormat mediaFormat) {
        int i = 0;
        try {
            if (mediaFormat.containsKey("rotation-degrees")) {
                i = mediaFormat.getInteger("rotation-degrees");
            } else {
                MediaMetadataRetriever mediaMetadataRetriever = new MediaMetadataRetriever();
                try {
                    mediaMetadataRetriever.setDataSource(this.mContext, this.mCurrentProcessingUri);
                    String extractMetadata = mediaMetadataRetriever.extractMetadata(24);
                    if (extractMetadata != null) {
                        i = Integer.parseInt(extractMetadata);
                    }
                } finally {
                    try {
                        mediaMetadataRetriever.release();
                    } catch (Exception unused) {
                    }
                }
            }
        } catch (Exception e) {
            LogUtil.e("PoseLandmarkerHelper5 resolveVideoRotation error: " + e.getMessage());
        }
        int i2 = ((i % 360) + 360) % 360;
        LogUtil.d("PoseLandmarkerHelper5 视频旋转角度=" + i2);
        return i2;
    }

    private Bitmap rotateBitmap(Bitmap bitmap, int i) {
        Bitmap bitmap2;
        Matrix matrix;
        try {
            matrix = new Matrix();
            matrix.postRotate(i);
            bitmap2 = bitmap;
        } catch (Exception e) {
            e = e;
            bitmap2 = bitmap;
        }
        try {
            return Bitmap.createBitmap(bitmap2, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        } catch (Exception e2) {
            e = e2;
            LogUtil.e("PoseLandmarkerHelper5 rotateBitmap error: " + e.getMessage());
            return bitmap2;
        }
    }

    private void startDetectionTask(long j) {
        startDetectionTask(j, true);
    }

    private void startDetectionTask(long j, boolean z) {
        startDetectionTask(j, z, false);
    }

    private void startDetectionTask(final long j, boolean z, final boolean z2) {
        if (this.singleExecutor.isShutdown()) {
            this.singleExecutor = Executors.newSingleThreadExecutor();
        }
        this.isReadyCallback = !z;
        final int incrementAndGet = this.taskGeneration.incrementAndGet();
        this.currentTask = this.singleExecutor.submit(new Runnable() { // from class: com.hismith.hismithapp.pose.PoseLandmarkerHelper5$$ExternalSyntheticLambda0
            @Override // java.lang.Runnable
            public final void run() {
                PoseLandmarkerHelper5.this.m619xc03e2799(j, incrementAndGet, z2);
            }
        });
    }

    private void trimCacheWindow(long j) {
        long j2 = this.mVideoDurationMs;
        if (j2 > 0) {
            long j3 = this.mPastKeepMs;
            long j4 = this.mFutureMaxMs;
            if (j2 <= j3 + j4) {
                return;
            }
            long min = Math.min(Math.max(this.mPlayReadyMs, (j4 + j) - j2), this.mVideoDurationMs);
            if (j - this.mPastKeepMs > min) {
                this.resultCache.subMap((boolean) Long.valueOf(min), false, (boolean) Long.valueOf(j - this.mPastKeepMs), false).clear();
            }
            this.resultCache.tailMap((ConcurrentSkipListMap<Long, PoseLandmarkerResult>) Long.valueOf(Math.max(j + this.mFutureMaxMs, this.mTaskLastCached)), false).clear();
        }
    }

    private Bitmap yuv420ToBitmap(ByteBuffer byteBuffer, int i, int i2, int i3) {
        int i4 = i * i2;
        int i5 = i4 / 2;
        int i6 = i4 + i5;
        if (byteBuffer.remaining() < i6) {
            LogUtil.e("PoseLandmarkerHelper5 yuv420ToBitmap: buffer too small " + byteBuffer.remaining() + " < " + i6);
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
        } else {
            if (i3 != 21) {
                LogUtil.e("PoseLandmarkerHelper5 yuv420ToBitmap: 未知色彩格式 " + i3 + "，跳过该帧");
                return null;
            }
            for (int i12 = 0; i12 < i5; i12 += 2) {
                int i13 = i4 + i12;
                byte b = bArr[i13];
                int i14 = i13 + 1;
                bArr2[i13] = bArr[i14];
                bArr2[i14] = b;
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
            LogUtil.e("PoseLandmarkerHelper5 yuv420ToBitmap error: " + e.getMessage());
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
                    LogUtil.d("PoseLandmarkerHelper5 detectVideoFileAsync: same video but cache incomplete, restart.");
                }
                LogUtil.d("PoseLandmarkerHelper5 detectVideoFileAsync: same video already loading/loaded, ignore.");
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
            this.mStorageWidth = 0;
            this.mStorageHeight = 0;
            this.mVideoRotation = 0;
            this.mTotalFrames = -1L;
            this.mCurrentPlayTime = 0L;
            this.mTaskLastCached = -1L;
            this.mTaskCachedFrames = 0;
            this.mTaskColorSkipped = 0;
            this.mTaskDetectErrors = 0;
            LogUtil.d("PoseLandmarkerHelper5 detectVideoFileAsync: start from 0");
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
        LogUtil.d("PoseLandmarkerHelper5 getCacheResult playTime=" + j + ", key=" + validTime + ", cacheSize=" + this.resultCache.size());
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

    /* JADX WARN: Removed duplicated region for block: B:120:0x022d A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:142:0x02b9  */
    /* JADX WARN: Removed duplicated region for block: B:151:0x03c1 A[ORIG_RETURN, RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:152:0x03b8 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:180:0x0223 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:193:0x03cd A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:199:? A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:200:0x03c5 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* renamed from: lambda$startDetectionTask$0$com-hismith-hismithapp-pose-PoseLandmarkerHelper5, reason: not valid java name */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    /* synthetic */ void m619xc03e2799(long j, int i, boolean z) {
        MediaExtractor mediaExtractor;
        int i2;
        MediaCodec mediaCodec;
        int i3;
        boolean z2;
        long j2;
        MediaExtractor mediaExtractor2;
        int i4;
        boolean z3;
        int dequeueOutputBuffer;
        boolean z4;
        long j3 = j;
        int i5 = 0;
        this.mTaskCachedFrames = 0;
        this.mTaskColorSkipped = 0;
        this.mTaskDetectErrors = 0;
        resetFrameStats();
        PoseLandmarker orCreatePoseLandmarker = getOrCreatePoseLandmarker(j);
        if (orCreatePoseLandmarker == null) {
            notifyComplete(i, false, 0, 1, "模型初始化失败", true, z);
            return;
        }
        MediaCodec mediaCodec2 = null;
        try {
            mediaExtractor = new MediaExtractor();
            try {
                try {
                    mediaExtractor.setDataSource(this.mContext, this.mCurrentProcessingUri, (Map<String, String>) null);
                    i2 = 0;
                    while (true) {
                        if (i2 >= mediaExtractor.getTrackCount()) {
                            i2 = -1;
                            break;
                        }
                        String string = mediaExtractor.getTrackFormat(i2).getString("mime");
                        if (string != null && string.startsWith("video/")) {
                            break;
                        } else {
                            i2++;
                        }
                    }
                } catch (Throwable th) {
                    th = th;
                }
            } catch (Exception e) {
                e = e;
            }
        } catch (Exception e2) {
            e = e2;
            mediaExtractor = null;
        } catch (Throwable th2) {
            th = th2;
            mediaExtractor = null;
        }
        if (i2 < 0) {
            try {
                LogUtil.e("PoseLandmarkerHelper5 startDetectionTask: no video track");
            } catch (Exception e3) {
                e = e3;
            }
            try {
                notifyComplete(i, false, 0, 2, "未找到视频轨道", true, z);
            } catch (Exception e4) {
                e = e4;
                e.printStackTrace();
                LogUtil.e("error:" + e.getMessage());
                notifyComplete(i, false, 0, 4, e.getMessage(), true, z);
                if (mediaCodec2 != null) {
                }
                if (mediaExtractor == null) {
                }
                try {
                    mediaExtractor.release();
                    return;
                } catch (Exception unused) {
                    return;
                }
            }
        } else {
            int i6 = i;
            mediaExtractor.selectTrack(i2);
            MediaFormat trackFormat = mediaExtractor.getTrackFormat(i2);
            this.mStorageWidth = trackFormat.getInteger(ViewHierarchyConstants.DIMENSION_WIDTH_KEY);
            this.mStorageHeight = trackFormat.getInteger(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY);
            int resolveVideoRotation = resolveVideoRotation(trackFormat);
            this.mVideoRotation = resolveVideoRotation;
            int i7 = this.mStorageWidth;
            this.mVideoWidth = i7;
            int i8 = this.mStorageHeight;
            this.mVideoHeight = i8;
            if (resolveVideoRotation == 90 || resolveVideoRotation == 270) {
                this.mVideoWidth = i8;
                this.mVideoHeight = i7;
            }
            long j4 = (trackFormat.containsKey("durationUs") ? trackFormat.getLong("durationUs") : -1L) / 1000;
            this.mVideoDurationMs = j4;
            long j5 = 1000;
            this.mTotalFrames = (j4 / this.mInferenceIntervalMs) + 1;
            VideoFrameCallback videoFrameCallback = this.mCallback;
            if (videoFrameCallback != null) {
                videoFrameCallback.onFrameRatio(this.mVideoWidth, this.mVideoHeight);
            }
            if (this.mVideoDurationMs > 0 && this.mVideoWidth != 0 && this.mVideoHeight != 0) {
                MediaCodec createDecoderByType = MediaCodec.createDecoderByType(trackFormat.getString("mime"));
                try {
                    try {
                        createDecoderByType.configure(trackFormat, (Surface) null, (MediaCrypto) null, 0);
                        createDecoderByType.start();
                        long max = Math.max((j3 - this.mSeekBackMs) * 1000, 0L);
                        LogUtil.d("PoseLandmarkerHelper5 跳转解码：目标=" + j3 + "ms, 实际seek=" + max + "us");
                        mediaExtractor.seekTo(max, 2);
                        MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
                        this.mTaskLastCached = -1L;
                        boolean z5 = false;
                        long j6 = -1;
                        boolean z6 = false;
                        while (!z5) {
                            try {
                                if (i6 != this.taskGeneration.get()) {
                                    break;
                                }
                                long max2 = Math.max(this.mCurrentPlayTime, j3);
                                if (this.mTaskLastCached > 0) {
                                    try {
                                        j2 = this.mTaskLastCached - max2;
                                        mediaExtractor2 = mediaExtractor;
                                    } catch (Exception e5) {
                                        e = e5;
                                        mediaCodec2 = createDecoderByType;
                                        e.printStackTrace();
                                        LogUtil.e("error:" + e.getMessage());
                                        notifyComplete(i, false, 0, 4, e.getMessage(), true, z);
                                        if (mediaCodec2 != null) {
                                        }
                                        if (mediaExtractor == null) {
                                        }
                                        mediaExtractor.release();
                                        return;
                                    } catch (Throwable th3) {
                                        th = th3;
                                        mediaCodec2 = createDecoderByType;
                                        if (mediaCodec2 != null) {
                                            try {
                                                mediaCodec2.stop();
                                            } catch (Exception unused2) {
                                            }
                                            try {
                                                mediaCodec2.release();
                                            } catch (Exception unused3) {
                                            }
                                        }
                                        if (mediaExtractor != null) {
                                            throw th;
                                        }
                                        try {
                                            mediaExtractor.release();
                                            throw th;
                                        } catch (Exception unused4) {
                                            throw th;
                                        }
                                    }
                                    try {
                                        long j7 = this.mFutureMaxMs;
                                        i4 = j2 > j7 ? 1 : i5;
                                        if (z) {
                                            long j8 = (max2 + j7) - this.mVideoDurationMs;
                                            if (j8 > 0 && this.mTaskLastCached >= j8) {
                                                i4 = 1;
                                            }
                                        }
                                    } catch (Exception e6) {
                                        e = e6;
                                        mediaCodec2 = createDecoderByType;
                                        mediaExtractor = mediaExtractor2;
                                        e.printStackTrace();
                                        LogUtil.e("error:" + e.getMessage());
                                        notifyComplete(i, false, 0, 4, e.getMessage(), true, z);
                                        if (mediaCodec2 != null) {
                                        }
                                        if (mediaExtractor == null) {
                                        }
                                        mediaExtractor.release();
                                        return;
                                    } catch (Throwable th4) {
                                        th = th4;
                                        mediaCodec2 = createDecoderByType;
                                        mediaExtractor = mediaExtractor2;
                                        if (mediaCodec2 != null) {
                                        }
                                        if (mediaExtractor != null) {
                                        }
                                    }
                                } else {
                                    mediaExtractor2 = mediaExtractor;
                                    i4 = i5;
                                }
                                if (i4 != 0) {
                                    try {
                                        LogUtil.d("PoseLandmarkerHelper5 缓存超前/回绕区已满，休眠等待播放推进，taskLastCached=" + this.mTaskLastCached + ", consumer=" + max2);
                                        Thread.sleep(50L);
                                        j3 = j;
                                        mediaExtractor = mediaExtractor2;
                                    } catch (InterruptedException unused5) {
                                        mediaCodec = createDecoderByType;
                                        i3 = i6;
                                        mediaExtractor = mediaExtractor2;
                                    }
                                } else {
                                    try {
                                        if (!z6) {
                                            try {
                                                int dequeueInputBuffer = createDecoderByType.dequeueInputBuffer(j5);
                                                if (dequeueInputBuffer >= 0) {
                                                    mediaExtractor = mediaExtractor2;
                                                    int readSampleData = mediaExtractor.readSampleData(createDecoderByType.getInputBuffer(dequeueInputBuffer), i5);
                                                    if (readSampleData < 0) {
                                                        MediaCodec mediaCodec3 = createDecoderByType;
                                                        try {
                                                            mediaCodec3.queueInputBuffer(dequeueInputBuffer, 0, 0, 0L, 4);
                                                            createDecoderByType = mediaCodec3;
                                                            z3 = true;
                                                            dequeueOutputBuffer = createDecoderByType.dequeueOutputBuffer(bufferInfo, 1000L);
                                                            if (dequeueOutputBuffer == -2) {
                                                                j3 = j;
                                                                i6 = i;
                                                                z6 = z3;
                                                                j5 = 1000;
                                                            } else {
                                                                if (dequeueOutputBuffer >= 0) {
                                                                    int i9 = (bufferInfo.flags & 4) != 0 ? 1 : i5;
                                                                    if (i9 == 0) {
                                                                        j5 = 1000;
                                                                        long j9 = bufferInfo.presentationTimeUs / 1000;
                                                                        long j10 = this.mInferenceIntervalMs;
                                                                        long j11 = (j9 / j10) * j10;
                                                                        if (j9 >= j) {
                                                                            if (this.resultCache.containsKey(Long.valueOf(j11))) {
                                                                                mediaCodec = createDecoderByType;
                                                                                z4 = true;
                                                                                i3 = i;
                                                                                if (j11 > this.mTaskLastCached) {
                                                                                    this.mTaskLastCached = j11;
                                                                                }
                                                                                if (j11 != j6) {
                                                                                    if (this.mCallback != null && i3 == this.taskGeneration.get()) {
                                                                                        this.mCallback.onCacheProgress(this.resultCache.size(), j11);
                                                                                    }
                                                                                    j6 = j11;
                                                                                }
                                                                            } else {
                                                                                z4 = true;
                                                                                try {
                                                                                    long processOutputFrame = processOutputFrame(createDecoderByType, dequeueOutputBuffer, trackFormat, j9, j11, j, orCreatePoseLandmarker, i);
                                                                                    mediaCodec = createDecoderByType;
                                                                                    i3 = i;
                                                                                    if (processOutputFrame > this.mTaskLastCached) {
                                                                                        this.mTaskLastCached = processOutputFrame;
                                                                                    }
                                                                                    if (processOutputFrame > j6) {
                                                                                        j6 = processOutputFrame;
                                                                                    }
                                                                                } catch (Exception e7) {
                                                                                    e = e7;
                                                                                    mediaCodec2 = createDecoderByType;
                                                                                    e.printStackTrace();
                                                                                    LogUtil.e("error:" + e.getMessage());
                                                                                    notifyComplete(i, false, 0, 4, e.getMessage(), true, z);
                                                                                    if (mediaCodec2 != null) {
                                                                                        try {
                                                                                            mediaCodec2.stop();
                                                                                        } catch (Exception unused6) {
                                                                                        }
                                                                                        try {
                                                                                            mediaCodec2.release();
                                                                                        } catch (Exception unused7) {
                                                                                        }
                                                                                    }
                                                                                    if (mediaExtractor == null) {
                                                                                        return;
                                                                                    }
                                                                                    mediaExtractor.release();
                                                                                    return;
                                                                                }
                                                                            }
                                                                            mediaCodec.releaseOutputBuffer(dequeueOutputBuffer, z4);
                                                                            if (i9 != 0) {
                                                                                z5 = z4;
                                                                            }
                                                                        } else {
                                                                            mediaCodec = createDecoderByType;
                                                                            z4 = true;
                                                                        }
                                                                    } else {
                                                                        mediaCodec = createDecoderByType;
                                                                        z4 = true;
                                                                        j5 = 1000;
                                                                    }
                                                                    i3 = i;
                                                                    mediaCodec.releaseOutputBuffer(dequeueOutputBuffer, z4);
                                                                    if (i9 != 0) {
                                                                    }
                                                                } else {
                                                                    mediaCodec = createDecoderByType;
                                                                    z2 = true;
                                                                    j5 = 1000;
                                                                    i3 = i;
                                                                    try {
                                                                        Thread.sleep(5L);
                                                                    } catch (InterruptedException unused8) {
                                                                    }
                                                                }
                                                                j3 = j;
                                                                i6 = i3;
                                                                createDecoderByType = mediaCodec;
                                                                z6 = z3;
                                                                i5 = 0;
                                                            }
                                                        } catch (Exception e8) {
                                                            e = e8;
                                                            mediaCodec2 = mediaCodec3;
                                                            e.printStackTrace();
                                                            LogUtil.e("error:" + e.getMessage());
                                                            notifyComplete(i, false, 0, 4, e.getMessage(), true, z);
                                                            if (mediaCodec2 != null) {
                                                            }
                                                            if (mediaExtractor == null) {
                                                            }
                                                            mediaExtractor.release();
                                                            return;
                                                        } catch (Throwable th5) {
                                                            th = th5;
                                                            mediaCodec2 = mediaCodec3;
                                                            if (mediaCodec2 != null) {
                                                            }
                                                            if (mediaExtractor != null) {
                                                            }
                                                        }
                                                    } else {
                                                        MediaCodec mediaCodec4 = createDecoderByType;
                                                        try {
                                                            mediaCodec4.queueInputBuffer(dequeueInputBuffer, 0, readSampleData, mediaExtractor.getSampleTime(), 0);
                                                            createDecoderByType = mediaCodec4;
                                                            mediaExtractor.advance();
                                                            z3 = z6;
                                                            dequeueOutputBuffer = createDecoderByType.dequeueOutputBuffer(bufferInfo, 1000L);
                                                            if (dequeueOutputBuffer == -2) {
                                                            }
                                                        } catch (Exception e9) {
                                                            e = e9;
                                                            createDecoderByType = mediaCodec4;
                                                            mediaCodec2 = createDecoderByType;
                                                            e.printStackTrace();
                                                            LogUtil.e("error:" + e.getMessage());
                                                            notifyComplete(i, false, 0, 4, e.getMessage(), true, z);
                                                            if (mediaCodec2 != null) {
                                                            }
                                                            if (mediaExtractor == null) {
                                                            }
                                                            mediaExtractor.release();
                                                            return;
                                                        } catch (Throwable th6) {
                                                            th = th6;
                                                            createDecoderByType = mediaCodec4;
                                                            mediaCodec2 = createDecoderByType;
                                                            if (mediaCodec2 != null) {
                                                            }
                                                            if (mediaExtractor != null) {
                                                            }
                                                        }
                                                    }
                                                }
                                            } catch (Exception e10) {
                                                e = e10;
                                                mediaExtractor = mediaExtractor2;
                                            } catch (Throwable th7) {
                                                th = th7;
                                                mediaExtractor = mediaExtractor2;
                                            }
                                        }
                                        dequeueOutputBuffer = createDecoderByType.dequeueOutputBuffer(bufferInfo, 1000L);
                                        if (dequeueOutputBuffer == -2) {
                                        }
                                    } catch (Exception e11) {
                                        e = e11;
                                        mediaCodec2 = createDecoderByType;
                                    }
                                    mediaExtractor = mediaExtractor2;
                                    z3 = z6;
                                }
                            } catch (Exception e12) {
                                e = e12;
                                mediaCodec2 = createDecoderByType;
                            }
                        }
                        mediaCodec = createDecoderByType;
                        i3 = i6;
                        z2 = true;
                        if (i3 == this.taskGeneration.get()) {
                            this.isIntact = this.mTotalFrames == ((long) this.resultCache.size()) ? z2 : false;
                            if (this.mTaskDetectErrors >= 20) {
                                notifyComplete(i3, false, this.resultCache.size(), 4, "推理连续异常，已中止", false, z);
                            } else if (this.mTaskCachedFrames != 0 || this.mTaskColorSkipped < 20) {
                                notifyComplete(i, true, this.resultCache.size(), 0, null, true, z);
                                if (this.mVideoDurationMs > this.mPastKeepMs + this.mFutureMaxMs && this.mTaskLastCached > 0) {
                                    LogUtil.d("PoseLandmarkerHelper5 缓存到片尾，开始从 0 回绕缓存");
                                    synchronized (this.taskLock) {
                                        startDetectionTask(0L, false, z2);
                                    }
                                }
                            } else {
                                notifyComplete(i, false, this.resultCache.size(), 5, "视频色彩格式不支持，无法进行姿态检测", false, z);
                            }
                        }
                        if (mediaCodec != null) {
                            try {
                                mediaCodec.stop();
                            } catch (Exception unused9) {
                            }
                            try {
                                mediaCodec.release();
                            } catch (Exception unused10) {
                            }
                        }
                    } catch (Exception e13) {
                        e = e13;
                    }
                    mediaExtractor.release();
                    return;
                } catch (Throwable th8) {
                    th = th8;
                }
            }
            notifyComplete(i, false, 0, 3, "视频元数据无效", true, z);
        }
        try {
            mediaExtractor.release();
        } catch (Exception unused11) {
        }
    }

    public boolean seekToDetectVideoAsync(long j) {
        synchronized (this.taskLock) {
            if (this.mCurrentProcessingUri == null) {
                LogUtil.e("PoseLandmarkerHelper5 seekToDetectVideoAsync: no video selected.");
                return false;
            }
            Long validTime = getValidTime(j);
            if (validTime != null && this.resultCache.containsKey(validTime) && isReadyAt(j)) {
                this.mCurrentPlayTime = validTime.longValue();
                if (this.mTaskLastCached > this.mFutureMaxMs + j) {
                    LogUtil.d("PoseLandmarkerHelper5 seek 回退但任务超前，后台重启补缓存，playTime=" + j + ", taskLastCached=" + this.mTaskLastCached);
                    cancelAsyncTask(false);
                    trimCacheWindow(j);
                    startDetectionTask(j);
                }
                LogUtil.d("PoseLandmarkerHelper5 seekToDetectVideoAsync: cache hit, playTime=" + j);
                return true;
            }
            LogUtil.d("PoseLandmarkerHelper5 seekToDetectVideoAsync: refill from " + j);
            cancelAsyncTask(false);
            this.mCurrentPlayTime = j;
            trimCacheWindow(j);
            this.mHandler.removeCallbacks(this.mSeekRunnable);
            this.mPendingPlayTime = j;
            this.mHandler.postDelayed(this.mSeekRunnable, SEEK_DEBOUNCE_DELAY_MS);
            return false;
        }
    }

    public void setCacheWindow(long j, long j2, long j3) {
        this.mPastKeepMs = Math.max(0L, j);
        long max = Math.max(0L, j3);
        this.mPlayReadyMs = max;
        this.mFutureMaxMs = Math.max(max, j2);
    }

    public void setSeekBackMs(long j) {
        this.mSeekBackMs = Math.max(0L, j);
    }
}
