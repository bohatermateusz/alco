package com.hismith.hismithapp.adapter;

import android.database.DataSetObservable;
import android.database.DataSetObserver;
import android.view.View;
import android.view.ViewGroup;

/* loaded from: classes.dex */
public abstract class XBaseAdapter {
    private DataSetObservable mObservable = new DataSetObservable();

    public abstract int getCount();

    public abstract View getView(int i, ViewGroup viewGroup);

    public void notifyDataSetChanged() {
        this.mObservable.notifyChanged();
    }

    public void registerDataSetObserver(DataSetObserver dataSetObserver) {
        this.mObservable.registerObserver(dataSetObserver);
    }

    public void unregisterDataSetObserver(DataSetObserver dataSetObserver) {
        this.mObservable.unregisterObserver(dataSetObserver);
    }
}
