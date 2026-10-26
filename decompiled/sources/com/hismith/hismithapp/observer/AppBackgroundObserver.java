package com.hismith.hismithapp.observer;

import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleObserver;
import androidx.lifecycle.OnLifecycleEvent;

/* loaded from: classes3.dex */
public class AppBackgroundObserver implements LifecycleObserver {
    @OnLifecycleEvent(Lifecycle.Event.ON_STOP)
    private void onAppBackground() {
        AppBackgroundUtil.isBackground = true;
        AppBackgroundUtil.dispatchBackgroundStatus(true);
    }

    @OnLifecycleEvent(Lifecycle.Event.ON_START)
    private void onAppForeground() {
        AppBackgroundUtil.isBackground = false;
        AppBackgroundUtil.dispatchBackgroundStatus(false);
    }
}
