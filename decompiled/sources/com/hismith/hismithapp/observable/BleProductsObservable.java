package com.hismith.hismithapp.observable;

import com.hismith.hismithapp.bluetooth.BleProduct;
import java.util.HashMap;
import java.util.Map;
import java.util.Observable;

/* loaded from: classes3.dex */
public class BleProductsObservable<T> extends Observable {
    private Map<String, BleProduct> bleProducts = new HashMap();

    public Map<String, BleProduct> getBleProducts() {
        return this.bleProducts;
    }

    public void put(String str, BleProduct bleProduct) {
        this.bleProducts.put(str, bleProduct);
        setChanged();
        notifyObservers();
    }
}
