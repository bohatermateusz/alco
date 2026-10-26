package com.hismith.hismithapp.audio;

/* loaded from: classes3.dex */
public enum RecordConfig {
    MIC(7),
    SAMPLE_RATE_INHZ(8000),
    CHANNEL_IN_CONFIG(16),
    CHANNEL_OUT_CONFIG(4),
    AUDIO_FORMAT(2);

    private int value;

    RecordConfig(int i) {
        this.value = i;
    }

    public int getValue() {
        return this.value;
    }
}
