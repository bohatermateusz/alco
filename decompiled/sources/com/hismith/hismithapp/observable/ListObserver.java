package com.hismith.hismithapp.observable;

import com.hismith.hismithapp.utils.LogUtil;
import java.util.Observable;
import java.util.Observer;

/* compiled from: BleProductsObservable.java */
/* loaded from: classes3.dex */
class ListObserver implements Observer {
    ListObserver() {
    }

    @Override // java.util.Observer
    public void update(Observable observable, Object obj) {
        if (observable instanceof BleProductsObservable) {
            LogUtil.e("list changed: " + ((BleProductsObservable) observable).getBleProducts().toString());
        }
    }
}
