package com.hismith.hismithapp.audio;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import com.hismith.hismithapp.audio.callback.TrackCallback;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

/* loaded from: classes3.dex */
public class TrackBase extends Thread implements BaseAudio {
    private AudioTrack audioTrack;
    private BlockingDeque<RTPPackage> blockingDeque;
    private int minTrackBufferSize;
    private TrackCallback trackCallback;
    private boolean isPlaying = true;
    private boolean isPark = true;

    public TrackBase() {
        init();
    }

    public void addMyTrackCallback(TrackCallback trackCallback) {
        this.trackCallback = trackCallback;
    }

    public boolean addPacket(RTPPackage rTPPackage) {
        return this.blockingDeque.offer(rTPPackage);
    }

    @Override // com.hismith.hismithapp.audio.BaseAudio
    public void begin() {
        LogUtil.e("startTrack");
        int state = this.audioTrack.getState();
        AudioTrack audioTrack = this.audioTrack;
        if (1 != state) {
            LogUtil.e("the audioTrack is uninitialized !  state = " + audioTrack.getState());
            return;
        }
        if (audioTrack != null) {
            audioTrack.play();
        }
        if (this.isPark) {
            this.isPark = false;
            LockSupport.unpark(this);
        }
    }

    @Override // com.hismith.hismithapp.audio.BaseAudio
    public void init() {
        this.minTrackBufferSize = 480;
        AudioTrack audioTrack = new AudioTrack(new AudioAttributes.Builder().setUsage(1).setContentType(2).build(), new AudioFormat.Builder().setSampleRate(RecordConfig.SAMPLE_RATE_INHZ.getValue()).setChannelMask(RecordConfig.CHANNEL_OUT_CONFIG.getValue()).setEncoding(RecordConfig.AUDIO_FORMAT.getValue()).build(), AudioTrack.getMinBufferSize(RecordConfig.SAMPLE_RATE_INHZ.getValue(), RecordConfig.CHANNEL_OUT_CONFIG.getValue(), RecordConfig.AUDIO_FORMAT.getValue()), 1, 0);
        this.audioTrack = audioTrack;
        LogUtil.e("TrackState:" + audioTrack.getState());
        this.blockingDeque = new LinkedBlockingDeque();
    }

    @Override // com.hismith.hismithapp.audio.BaseAudio
    public void over() {
        LogUtil.e("stopTrack");
        AudioTrack audioTrack = this.audioTrack;
        if (audioTrack != null) {
            audioTrack.stop();
        }
        if (this.isPark) {
            return;
        }
        this.isPark = true;
    }

    @Override // com.hismith.hismithapp.audio.BaseAudio
    public void release() {
        LogUtil.e("release");
        AudioTrack audioTrack = this.audioTrack;
        if (audioTrack != null) {
            this.isPlaying = false;
            audioTrack.release();
            this.audioTrack = null;
            LockSupport.unpark(this);
        }
    }

    @Override // java.lang.Thread, java.lang.Runnable
    public void run() {
        LogUtil.e("run");
        LogUtil.e("TrackThread running -------------------");
        while (this.isPlaying) {
            try {
                LogUtil.e("Playing ----------------------");
                if (this.isPark) {
                    if (this.blockingDeque.size() > 0) {
                        this.blockingDeque.clear();
                        LogUtil.w("blockingDeque.size() = " + this.blockingDeque.size());
                    }
                    LockSupport.park();
                }
                RTPPackage poll = this.blockingDeque.poll(2L, TimeUnit.SECONDS);
                LogUtil.w("blockingDeque.size() = " + this.blockingDeque.size());
                TrackCallback trackCallback = this.trackCallback;
                if (trackCallback != null) {
                    trackCallback.catchCount(this.blockingDeque.size());
                }
                if (poll != null) {
                    byte[] opusData = poll.getOpusData();
                    LogUtil.w("recording ---- decode  >>");
                    LogUtil.w("shortBuffer ： " + opusData.length);
                    if (opusData != null) {
                        this.audioTrack.write(opusData, 0, opusData.length);
                    }
                } else {
                    TrackCallback trackCallback2 = this.trackCallback;
                    if (trackCallback2 != null) {
                        trackCallback2.playTimeOut();
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                LogUtil.e("Playing ERROR:" + e);
            }
        }
    }
}
