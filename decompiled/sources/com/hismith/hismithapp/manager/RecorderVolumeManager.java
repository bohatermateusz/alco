package com.hismith.hismithapp.manager;

import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Message;
import com.hismith.hismithapp.utils.FileUtil;
import com.hismith.hismithapp.utils.LogUtil;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

/* loaded from: classes3.dex */
public class RecorderVolumeManager {
    private boolean isRecording;
    private String mFileName;
    private OnMicStatusListener onMicStatusListener;
    private long startTime;
    private long timeInterval;
    private MediaRecorder mRecorder = null;
    private final Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.manager.RecorderVolumeManager.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1) {
                return false;
            }
            RecorderVolumeManager.this.updateMicStatus();
            return false;
        }
    });
    private final Runnable mUpdateMicStatusTimer = new Runnable() { // from class: com.hismith.hismithapp.manager.RecorderVolumeManager.2
        @Override // java.lang.Runnable
        public void run() {
            RecorderVolumeManager.this.updateMicStatus();
        }
    };
    private final int BASE = 1;
    private final int SPACE = 100;

    public interface OnMicStatusListener {
        void onMicAmplitude(double d);
    }

    public RecorderVolumeManager() {
        this.mFileName = null;
        this.mFileName = FileUtil.getCacheFilePath("tempAudio1.mp4");
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

    /* JADX INFO: Access modifiers changed from: private */
    public void updateMicStatus() {
        if (this.mRecorder != null) {
            double maxAmplitude = r0.getMaxAmplitude() / 1.0d;
            double log10 = maxAmplitude > 1.0d ? Math.log10(maxAmplitude) * 20.0d : 0.0d;
            LogUtil.e("updateMicStatus db:" + log10);
            OnMicStatusListener onMicStatusListener = this.onMicStatusListener;
            if (onMicStatusListener != null) {
                onMicStatusListener.onMicAmplitude(log10);
            }
            this.mHandler.sendEmptyMessageDelayed(1, 100L);
        }
    }

    public synchronized void cancelRecording() {
        if (this.mHandler.hasMessages(1)) {
            this.mHandler.removeMessages(1);
        }
        MediaRecorder mediaRecorder = this.mRecorder;
        if (mediaRecorder != null) {
            try {
                mediaRecorder.release();
                this.mRecorder = null;
            } catch (Exception e) {
                e.printStackTrace();
            }
            new File(this.mFileName).deleteOnExit();
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

    public boolean isRecording() {
        return this.isRecording;
    }

    public void setOnMicStatusListener(OnMicStatusListener onMicStatusListener) {
        this.onMicStatusListener = onMicStatusListener;
    }

    public void startRecording() {
        if (this.mFileName == null) {
            return;
        }
        if (this.isRecording) {
            this.mRecorder.release();
            this.mRecorder = null;
        }
        LogUtil.e("Voice OnTouch mFileName:" + this.mFileName);
        try {
            MediaRecorder mediaRecorder = new MediaRecorder();
            this.mRecorder = mediaRecorder;
            mediaRecorder.setAudioSource(1);
            this.mRecorder.setOutputFormat(1);
            this.mRecorder.setAudioEncoder(1);
            this.mRecorder.setOutputFile(this.mFileName);
            this.startTime = System.currentTimeMillis();
            this.mRecorder.prepare();
            this.mRecorder.start();
            this.isRecording = true;
            updateMicStatus();
        } catch (Exception unused) {
            LogUtil.e("Voice prepare() failed");
        }
    }

    public void startRecording(String str) {
        this.mFileName = FileUtil.getCacheFilePath(str);
        startRecording();
    }

    public void stopRecording() {
        if (this.mHandler.hasMessages(1)) {
            this.mHandler.removeMessages(1);
        }
        if (this.mFileName == null || this.startTime == 0) {
            return;
        }
        long currentTimeMillis = System.currentTimeMillis() - this.startTime;
        this.timeInterval = currentTimeMillis;
        if (currentTimeMillis > 1000) {
            try {
                this.mRecorder.stop();
            } catch (Exception unused) {
                LogUtil.e("Voice release() failed");
                return;
            }
        }
        this.mRecorder.release();
        this.mRecorder = null;
        this.isRecording = false;
    }
}
