package com.hismith.hismithapp.listener;

import android.view.View;

/* loaded from: classes3.dex */
public abstract class CustomClickListener implements View.OnClickListener {
    private long mLastClickTime;
    private long timeInterval;

    public CustomClickListener() {
        this.timeInterval = 1000L;
    }

    public CustomClickListener(long j) {
        this.timeInterval = j;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        long currentTimeMillis = System.currentTimeMillis();
        if (currentTimeMillis - this.mLastClickTime > this.timeInterval) {
            onSingleClick(view);
            this.mLastClickTime = currentTimeMillis;
        }
    }

    protected abstract void onSingleClick(View view);
}
