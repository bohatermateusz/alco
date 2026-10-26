package com.hismith.hismithapp.manager;

import android.app.Activity;
import android.content.Context;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.util.Base64;
import androidx.media3.extractor.AacUtil;
import com.hismith.hismithapp.http.LomiSocketManager;
import com.hismith.hismithapp.model.MessageVoiceModel;
import com.hismith.hismithapp.utils.LogUtil;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.concurrent.LinkedBlockingQueue;

/* loaded from: classes3.dex */
public class VoiceCallManager3 {
    private static final String TAG = "VoiceCallManager3";
    private static VoiceCallManager3 instance;
    private final int VAD_TIMEOUT;
    private Context appContext;
    private final int bufferSize;
    private Callback callback;
    private boolean inSpeaking;
    private long lastVoiceTime;
    private LinkedBlockingQueue<byte[]> playQueue;
    private Thread playThread;
    private AudioTrack player;
    private byte[] recordBuffer;
    private ByteArrayOutputStream recordOutput;
    private Thread recordThread;
    private AudioRecord recorder;
    private boolean isRecording = false;
    private boolean isPlaying = false;
    private boolean isInCallMode = false;
    private boolean micMuted = false;
    private boolean speakerMuted = false;
    private int playerSampleRate = 22050;
    private int playerBufferSize = AudioRecord.getMinBufferSize(22050, 16, 2);
    private final int sampleRate = AacUtil.AAC_HE_V1_MAX_RATE_BYTES_PER_SECOND;

    public interface Callback {
        void onPlayStateChanged(boolean z);

        void onRecordStateChanged(boolean z);
    }

    public VoiceCallManager3() {
        int minBufferSize = AudioRecord.getMinBufferSize(AacUtil.AAC_HE_V1_MAX_RATE_BYTES_PER_SECOND, 16, 2);
        this.bufferSize = minBufferSize;
        this.playQueue = new LinkedBlockingQueue<>();
        this.lastVoiceTime = 0L;
        this.inSpeaking = false;
        this.VAD_TIMEOUT = 1000;
        this.recordBuffer = new byte[minBufferSize * 2];
        this.recordOutput = new ByteArrayOutputStream();
    }

    private boolean detectVoice(byte[] bArr) {
        long j = 0;
        for (int i = 0; i < bArr.length; i += 2) {
            j += Math.abs((int) ((short) ((bArr[i + 1] << 8) | (bArr[i] & 255))));
        }
        return j / ((long) (bArr.length / 2)) > 300;
    }

    public static VoiceCallManager3 getInstance() {
        VoiceCallManager3 voiceCallManager3 = instance;
        if (voiceCallManager3 != null) {
            return voiceCallManager3;
        }
        VoiceCallManager3 voiceCallManager32 = new VoiceCallManager3();
        instance = voiceCallManager32;
        return voiceCallManager32;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void recordLoop() {
        while (this.isRecording) {
            AudioRecord audioRecord = this.recorder;
            byte[] bArr = this.recordBuffer;
            int read = audioRecord.read(bArr, 0, bArr.length);
            if (!this.micMuted && read > 0) {
                byte[] copyOf = Arrays.copyOf(this.recordBuffer, read);
                if (detectVoice(copyOf)) {
                    this.lastVoiceTime = System.currentTimeMillis();
                    this.inSpeaking = true;
                    this.recordOutput.write(copyOf, 0, copyOf.length);
                } else if (this.inSpeaking && System.currentTimeMillis() - this.lastVoiceTime > 1000) {
                    byte[] byteArray = this.recordOutput.toByteArray();
                    if (byteArray.length > 0) {
                        sendBase64(byteArray);
                    }
                    this.recordOutput.reset();
                    this.inSpeaking = false;
                }
            }
        }
    }

    private void sendBase64(byte[] bArr) {
        LomiSocketManager.getInstance().lomiVoice(Base64.encodeToString(bArr, 2));
    }

    public void enableCallMode(Activity activity) {
        LogUtil.e("enableCallMode()");
        AudioManager audioManager = (AudioManager) activity.getSystemService("audio");
        audioManager.setMode(3);
        audioManager.setSpeakerphoneOn(!this.speakerMuted);
        this.isInCallMode = true;
    }

    public void init(Context context, Callback callback) {
        this.appContext = context;
        this.callback = callback;
    }

    /* renamed from: lambda$playVoice$0$com-hismith-hismithapp-manager-VoiceCallManager3, reason: not valid java name */
    /* synthetic */ void m611xe9083b76() {
        while (this.isPlaying) {
            try {
                byte[] take = this.playQueue.take();
                this.player.write(take, 0, take.length);
            } catch (Exception unused) {
            }
        }
    }

    /* renamed from: lambda$playVoice$1$com-hismith-hismithapp-manager-VoiceCallManager3, reason: not valid java name */
    /* synthetic */ void m612xdab1e195() {
        while (this.isPlaying) {
            try {
                byte[] take = this.playQueue.take();
                this.player.write(take, 0, take.length);
            } catch (Exception unused) {
            }
        }
    }

    public void playVoice(MessageVoiceModel messageVoiceModel) {
        if (this.isRecording) {
            stopPlay();
        }
        if (this.player == null || this.playerSampleRate != messageVoiceModel.getRate()) {
            int rate = messageVoiceModel.getRate();
            this.playerSampleRate = rate;
            this.playerBufferSize = AudioRecord.getMinBufferSize(rate, 16, 2);
            AudioTrack audioTrack = new AudioTrack(0, this.playerSampleRate, 4, 2, this.playerBufferSize * 2, 1);
            this.player = audioTrack;
            audioTrack.setVolume(this.speakerMuted ? 0.0f : 1.0f);
        }
        if (!this.isPlaying) {
            this.isPlaying = true;
            this.player.play();
            Callback callback = this.callback;
            if (callback != null) {
                callback.onPlayStateChanged(true);
            }
            Thread thread = new Thread(new Runnable() { // from class: com.hismith.hismithapp.manager.VoiceCallManager3$$ExternalSyntheticLambda0
                @Override // java.lang.Runnable
                public final void run() {
                    VoiceCallManager3.this.m611xe9083b76();
                }
            });
            this.playThread = thread;
            thread.start();
        }
        this.playQueue.offer(Base64.decode(messageVoiceModel.getDat(), 2));
    }

    public void playVoice(byte[] bArr) {
        if (this.isRecording) {
            stopPlay();
        }
        if (this.player == null) {
            AudioTrack audioTrack = new AudioTrack(0, this.playerSampleRate, 4, 2, this.bufferSize * 2, 1);
            this.player = audioTrack;
            audioTrack.setVolume(this.speakerMuted ? 0.0f : 1.0f);
        }
        if (!this.isPlaying) {
            this.isPlaying = true;
            this.player.play();
            Callback callback = this.callback;
            if (callback != null) {
                callback.onPlayStateChanged(true);
            }
            Thread thread = new Thread(new Runnable() { // from class: com.hismith.hismithapp.manager.VoiceCallManager3$$ExternalSyntheticLambda2
                @Override // java.lang.Runnable
                public final void run() {
                    VoiceCallManager3.this.m612xdab1e195();
                }
            });
            this.playThread = thread;
            thread.start();
        }
        this.playQueue.offer(bArr);
    }

    public void release() {
        LogUtil.e("VoiceCallManager3 release()");
        try {
            this.isRecording = false;
            AudioRecord audioRecord = this.recorder;
            if (audioRecord != null) {
                audioRecord.stop();
                this.recorder.release();
                this.recorder = null;
            }
        } catch (Exception unused) {
        }
        try {
            this.isPlaying = false;
            this.playQueue.clear();
            AudioTrack audioTrack = this.player;
            if (audioTrack != null) {
                audioTrack.stop();
                this.player.flush();
                this.player.release();
                this.player = null;
            }
        } catch (Exception unused2) {
        }
        try {
            Thread thread = this.recordThread;
            if (thread != null) {
                thread.interrupt();
                this.recordThread = null;
            }
            Thread thread2 = this.playThread;
            if (thread2 != null) {
                thread2.interrupt();
                this.playThread = null;
            }
        } catch (Exception unused3) {
        }
        this.inSpeaking = false;
        this.lastVoiceTime = 0L;
        this.recordOutput.reset();
        try {
            Context context = this.appContext;
            if (context != null) {
                ((AudioManager) context.getSystemService("audio")).setMode(0);
            }
        } catch (Exception unused4) {
        }
        LogUtil.e("VoiceCallManager3 已完全释放");
    }

    public void setMicMute(boolean z) {
        this.micMuted = z;
    }

    public void setSpeakerMute(boolean z) {
        this.speakerMuted = z;
        AudioTrack audioTrack = this.player;
        if (audioTrack != null) {
            audioTrack.setVolume(z ? 0.0f : 1.0f);
        }
    }

    public void startRecord() {
        if (this.isRecording) {
            return;
        }
        AudioRecord audioRecord = new AudioRecord(7, AacUtil.AAC_HE_V1_MAX_RATE_BYTES_PER_SECOND, 16, 2, this.bufferSize);
        this.recorder = audioRecord;
        audioRecord.startRecording();
        this.isRecording = true;
        Callback callback = this.callback;
        if (callback != null) {
            callback.onRecordStateChanged(true);
        }
        this.recordOutput.reset();
        Thread thread = this.recordThread;
        if (thread == null || !thread.isAlive()) {
            Thread thread2 = new Thread(new Runnable() { // from class: com.hismith.hismithapp.manager.VoiceCallManager3$$ExternalSyntheticLambda1
                @Override // java.lang.Runnable
                public final void run() {
                    VoiceCallManager3.this.recordLoop();
                }
            });
            this.recordThread = thread2;
            thread2.start();
        }
    }

    public void stopPlay() {
        this.isPlaying = false;
        this.playQueue.clear();
        AudioTrack audioTrack = this.player;
        if (audioTrack != null) {
            audioTrack.pause();
            this.player.flush();
        }
        Callback callback = this.callback;
        if (callback != null) {
            callback.onPlayStateChanged(false);
        }
    }

    public void stopRecordAndSend() {
        if (this.isRecording) {
            this.isRecording = false;
            Callback callback = this.callback;
            if (callback != null) {
                callback.onRecordStateChanged(false);
            }
            AudioRecord audioRecord = this.recorder;
            if (audioRecord != null) {
                audioRecord.stop();
                this.recorder.release();
                this.recorder = null;
            }
            byte[] byteArray = this.recordOutput.toByteArray();
            if (byteArray.length > 0) {
                sendBase64(byteArray);
            }
        }
    }
}
