package com.hismith.hismithapp.manager;

import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;

/* loaded from: classes3.dex */
public abstract class DownLoadObserver implements Observer<DownloadInfo> {
    protected Disposable d;
    protected DownloadInfo downloadInfo;

    @Override // io.reactivex.Observer
    public void onError(Throwable th) {
        th.printStackTrace();
    }

    @Override // io.reactivex.Observer
    public void onNext(DownloadInfo downloadInfo) {
        this.downloadInfo = downloadInfo;
    }

    @Override // io.reactivex.Observer
    public void onSubscribe(Disposable disposable) {
        this.d = disposable;
    }
}
