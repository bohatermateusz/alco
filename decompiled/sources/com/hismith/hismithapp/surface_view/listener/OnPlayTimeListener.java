package com.hismith.hismithapp.surface_view.listener;

/* loaded from: classes3.dex */
public interface OnPlayTimeListener {
    void onComplete();

    void onFunData(String str, String str2);

    void onLinePause();

    void onPlayTime(int i);

    void onPositionRangeChange(int i, int i2);

    void onSmoothChange(int i);

    void onSpeedChange(int i);

    void onVibrateChange(int i);
}
