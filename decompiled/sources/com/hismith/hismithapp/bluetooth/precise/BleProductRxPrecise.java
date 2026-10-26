package com.hismith.hismithapp.bluetooth.precise;

import android.util.Log;
import io.reactivex.Observable;
import io.reactivex.ObservableSource;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.functions.Consumer;
import io.reactivex.functions.Function;
import io.reactivex.schedulers.Schedulers;
import io.reactivex.subjects.PublishSubject;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/* loaded from: classes3.dex */
public class BleProductRxPrecise {
    private static final String TAG = "BleProduct";
    private final PublishSubject<String> dataSubject = PublishSubject.create();
    private final CompositeDisposable disposables = new CompositeDisposable();
    private final AtomicLong lastExecuteTime = new AtomicLong(0);

    public BleProductRxPrecise() {
        setupProcessing();
    }

    private void sendToBle(String str) {
        Log.d(TAG, "发送BLE数据: " + str + ", 时间: " + System.currentTimeMillis());
    }

    private void setupProcessing() {
        this.disposables.add(this.dataSubject.observeOn(Schedulers.io()).concatMap(new Function() { // from class: com.hismith.hismithapp.bluetooth.precise.BleProductRxPrecise$$ExternalSyntheticLambda1
            @Override // io.reactivex.functions.Function
            public final Object apply(Object obj) {
                return BleProductRxPrecise.this.m582x287d309d((String) obj);
            }
        }).subscribe(new Consumer() { // from class: com.hismith.hismithapp.bluetooth.precise.BleProductRxPrecise$$ExternalSyntheticLambda2
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                Log.d(BleProductRxPrecise.TAG, "数据处理完成: " + ((String) obj));
            }
        }, new Consumer() { // from class: com.hismith.hismithapp.bluetooth.precise.BleProductRxPrecise$$ExternalSyntheticLambda3
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                Log.e(BleProductRxPrecise.TAG, "处理错误", (Throwable) obj);
            }
        }));
    }

    public void cleanup() {
        this.disposables.dispose();
        Log.d(TAG, "资源已清理");
    }

    /* renamed from: lambda$setupProcessing$0$com-hismith-hismithapp-bluetooth-precise-BleProductRxPrecise, reason: not valid java name */
    /* synthetic */ void m581x943ec0fe(String str) throws Exception {
        this.lastExecuteTime.set(System.currentTimeMillis());
        sendToBle(str);
    }

    /* renamed from: lambda$setupProcessing$1$com-hismith-hismithapp-bluetooth-precise-BleProductRxPrecise, reason: not valid java name */
    /* synthetic */ ObservableSource m582x287d309d(String str) throws Exception {
        long currentTimeMillis = System.currentTimeMillis();
        long j = this.lastExecuteTime.get();
        long j2 = currentTimeMillis - j;
        Observable just = Observable.just(str);
        if (j2 < 100 && j > 0) {
            just = just.delay(100 - j2, TimeUnit.MILLISECONDS, Schedulers.io());
        }
        return just.doOnNext(new Consumer() { // from class: com.hismith.hismithapp.bluetooth.precise.BleProductRxPrecise$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                BleProductRxPrecise.this.m581x943ec0fe((String) obj);
            }
        });
    }

    public void writeData(String str) {
        if (str == null || str.trim().isEmpty()) {
            return;
        }
        this.dataSubject.onNext(str);
    }
}
