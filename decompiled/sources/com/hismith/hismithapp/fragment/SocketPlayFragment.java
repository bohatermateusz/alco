package com.hismith.hismithapp.fragment;

import com.hismith.hismithapp.model.SocketProductModel;

/* loaded from: classes3.dex */
public abstract class SocketPlayFragment extends LazyLoadFragment {
    public void initBleAAData(String str, String str2) {
    }

    public void initBleCCData(String str, String str2, String str3) {
    }

    public void initProductData() {
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
    }

    public abstract void onNotify(String str);

    public void onSetSafeMax(int i) {
    }

    public void onSetVibrateSafeMax(int i) {
    }

    public void updateSocketProductModel(SocketProductModel socketProductModel) {
    }
}
