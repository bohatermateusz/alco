package com.hismith.hismithapp.utils;

import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import com.hismith.hismithapp.utils.RecorderUtil;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

/* loaded from: classes3.dex */
public class RecorderUtil {
    private boolean isRecording;
    private String mFileName;
    private OnMicStatusListener onMicStatusListener;
    private long startTime;
    private long timeInterval;
    private MediaRecorder mRecorder = null;
    private AudioRecord audioRecord = null;
    private Thread volumeThread = null;
    private volatile boolean isVolumeMonitoring = false;
    private int audioSource = 1;
    private int sampleRateInHz = 44100;
    private int channelConfig = 16;
    private int audioFormat = 2;
    private int bufferSizeInBytes = 0;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.utils.RecorderUtil$$ExternalSyntheticLambda0
        @Override // android.os.Handler.Callback
        public final boolean handleMessage(Message message) {
            return RecorderUtil.lambda$new$0(message);
        }
    });

    /* renamed from: com.hismith.hismithapp.utils.RecorderUtil$1, reason: invalid class name */
    class AnonymousClass1 implements Runnable {
        AnonymousClass1() {
        }

        /* renamed from: lambda$run$0$com-hismith-hismithapp-utils-RecorderUtil$1, reason: not valid java name */
        /* synthetic */ void m625lambda$run$0$comhismithhismithapputilsRecorderUtil$1(double d) {
            RecorderUtil.this.onMicStatusListener.onMicAmplitude(d);
        }

        @Override // java.lang.Runnable
        public void run() {
            int i = RecorderUtil.this.bufferSizeInBytes;
            byte[] bArr = new byte[i];
            long j = 0;
            double d = 0.0d;
            while (RecorderUtil.this.isVolumeMonitoring && RecorderUtil.this.audioRecord != null) {
                int read = RecorderUtil.this.audioRecord.read(bArr, 0, i);
                if (read > 0) {
                    double calculateRMS = RecorderUtil.this.calculateRMS(bArr, read);
                    if (calculateRMS > d) {
                        d = calculateRMS;
                    }
                    long currentTimeMillis = System.currentTimeMillis();
                    if (currentTimeMillis - j >= 100) {
                        final double log10 = Math.log10(d) * 26.0d;
                        LogUtil.e("updateMicStatus rms(max):" + d + " db:" + log10);
                        if (RecorderUtil.this.onMicStatusListener != null) {
                            RecorderUtil.this.mainHandler.post(new Runnable() { // from class: com.hismith.hismithapp.utils.RecorderUtil$1$$ExternalSyntheticLambda0
                                @Override // java.lang.Runnable
                                public final void run() {
                                    RecorderUtil.AnonymousClass1.this.m625lambda$run$0$comhismithhismithapputilsRecorderUtil$1(log10);
                                }
                            });
                        }
                        d = 0.0d;
                        j = currentTimeMillis;
                    }
                }
            }
        }
    }

    public interface OnMicStatusListener {
        void onMicAmplitude(double d);
    }

    public RecorderUtil() {
        this.mFileName = null;
        this.mFileName = FileUtil.getCacheFilePath("tempAudio.mp4");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public double calculateRMS(byte[] bArr, int i) {
        double d = 0.0d;
        for (int i2 = 0; i2 < i; i2 += 2) {
            int i3 = (bArr[i2 + 1] << 8) | (bArr[i2] & 255);
            d += i3 * i3;
        }
        return Math.sqrt(d / (i / 2));
    }

    private void initAudioRecord() {
        int minBufferSize = AudioRecord.getMinBufferSize(this.sampleRateInHz, this.channelConfig, this.audioFormat);
        this.bufferSizeInBytes = minBufferSize;
        if (minBufferSize == -2 || minBufferSize == -1) {
            LogUtil.e("AudioRecord getMinBufferSize error");
            return;
        }
        AudioRecord audioRecord = new AudioRecord(this.audioSource, this.sampleRateInHz, this.channelConfig, this.audioFormat, this.bufferSizeInBytes);
        this.audioRecord = audioRecord;
        if (audioRecord.getState() != 1) {
            LogUtil.e("AudioRecord initialize failed");
            this.audioRecord = null;
        } else {
            this.audioRecord.startRecording();
            this.isVolumeMonitoring = true;
        }
    }

    static /* synthetic */ boolean lambda$new$0(Message message) {
        return false;
    }

    private static byte[] readFile(File file) throws IOException {
        RandomAccessFile randomAccessFile = new RandomAccessFile(file, "r");
        try {
            long length = randomAccessFile.length();
            int i = (int) length;
            if (i != length) {
                throw new IOException("File size >= 2 GB");
            }
            byte[] bArr = new byte[i];
            randomAccessFile.readFully(bArr);
            return bArr;
        } finally {
            randomAccessFile.close();
        }
    }

    private void releaseAudioRecord() {
        AudioRecord audioRecord = this.audioRecord;
        if (audioRecord != null) {
            try {
                audioRecord.stop();
                this.audioRecord.release();
            } catch (Exception unused) {
            }
            this.audioRecord = null;
        }
    }

    private void releaseRecorder() {
        MediaRecorder mediaRecorder = this.mRecorder;
        if (mediaRecorder != null) {
            try {
                mediaRecorder.release();
            } catch (Exception unused) {
            }
            this.mRecorder = null;
        }
    }

    private void startVolumeMonitoring() {
        if (this.audioRecord == null) {
            return;
        }
        Thread thread = new Thread(new AnonymousClass1());
        this.volumeThread = thread;
        thread.start();
    }

    private void stopVolumeMonitoring() {
        this.isVolumeMonitoring = false;
        Thread thread = this.volumeThread;
        if (thread != null) {
            try {
                thread.join(500L);
            } catch (InterruptedException unused) {
            }
            this.volumeThread = null;
        }
        releaseAudioRecord();
    }

    public synchronized void cancelRecording() {
        if (this.mHandler.hasMessages(1)) {
            this.mHandler.removeMessages(1);
        }
        stopVolumeMonitoring();
        MediaRecorder mediaRecorder = this.mRecorder;
        if (mediaRecorder != null) {
            try {
                mediaRecorder.release();
            } catch (Exception e) {
                e.printStackTrace();
            }
            this.mRecorder = null;
        }
        File file = new File(this.mFileName);
        if (file.exists()) {
            file.delete();
        }
        this.isRecording = false;
    }

    public byte[] getDate() {
        if (this.mFileName == null) {
            return null;
        }
        try {
            return readFile(new File(this.mFileName));
        } catch (IOException e) {
            LogUtil.e("Voice read file error" + e);
            return null;
        }
    }

    public String getFilePath() {
        return this.mFileName;
    }

    public long getTimeInterval() {
        return this.timeInterval / 1000;
    }

    public void setOnMicStatusListener(OnMicStatusListener onMicStatusListener) {
        this.onMicStatusListener = onMicStatusListener;
    }

    public void startRecording() {
        if (this.mFileName == null) {
            return;
        }
        if (this.isRecording) {
            releaseRecorder();
            releaseAudioRecord();
        }
        LogUtil.e("Voice OnTouch mFileName:" + this.mFileName);
        try {
            MediaRecorder mediaRecorder = new MediaRecorder();
            this.mRecorder = mediaRecorder;
            mediaRecorder.setAudioSource(1);
            this.mRecorder.setOutputFormat(2);
            this.mRecorder.setAudioEncoder(3);
            this.mRecorder.setOutputFile(this.mFileName);
            this.mRecorder.prepare();
            this.mRecorder.start();
            this.startTime = System.currentTimeMillis();
            this.isRecording = true;
            initAudioRecord();
            startVolumeMonitoring();
        } catch (Exception e) {
            LogUtil.e("Voice prepare() failed: " + e.getMessage());
            releaseRecorder();
            releaseAudioRecord();
            this.isRecording = false;
        }
    }

    public void startRecording(String str) {
        this.mFileName = FileUtil.getCacheFilePath(str);
        startRecording();
    }

    public void stopRecording() {
        MediaRecorder mediaRecorder;
        if (this.mHandler.hasMessages(1)) {
            this.mHandler.removeMessages(1);
        }
        if (this.mFileName == null || this.startTime == 0) {
            return;
        }
        this.timeInterval = System.currentTimeMillis() - this.startTime;
        stopVolumeMonitoring();
        try {
            if (this.timeInterval > 1000 && (mediaRecorder = this.mRecorder) != null) {
                mediaRecorder.stop();
            }
            releaseRecorder();
        } catch (Exception e) {
            LogUtil.e("Voice stop() failed: " + e.getMessage());
        }
        this.isRecording = false;
    }
}
