package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public abstract class CommAdapter<T> extends XBaseAdapter {
    protected Context mContext;
    protected List<T> mData;
    protected LayoutInflater mInflater;
    protected int mLayoutId;

    public CommAdapter(Context context, int i) {
        this(context, null, i);
    }

    public CommAdapter(Context context, List<T> list, int i) {
        this.mContext = context;
        this.mData = list == null ? new ArrayList() : new ArrayList(list);
        this.mLayoutId = i;
        this.mInflater = LayoutInflater.from(this.mContext);
    }

    public void add(T t) {
        this.mData.add(t);
        notifyDataSetChanged();
    }

    public void addAll(List<T> list) {
        this.mData.addAll(list);
        notifyDataSetChanged();
    }

    public void addFirst(T t) {
        this.mData.add(0, t);
        notifyDataSetChanged();
    }

    public void changeAll(List<T> list) {
        this.mData.clear();
        this.mData.addAll(list);
    }

    public void clear() {
        this.mData.clear();
        notifyDataSetChanged();
    }

    public boolean contains(T t) {
        return this.mData.contains(t);
    }

    public abstract void convert(XQuickViewHolder xQuickViewHolder, T t, int i);

    @Override // com.hismith.hismithapp.adapter.XBaseAdapter
    public int getCount() {
        return this.mData.size();
    }

    public List<T> getData() {
        return this.mData;
    }

    @Override // com.hismith.hismithapp.adapter.XBaseAdapter
    public View getView(int i, ViewGroup viewGroup) {
        View inflate = this.mInflater.inflate(this.mLayoutId, viewGroup, false);
        convert(new XQuickViewHolder(inflate), this.mData.get(i), i);
        return inflate;
    }

    public boolean isLast(int i) {
        List<T> list = this.mData;
        return list != null && list.size() - 1 == i;
    }

    public void remove(int i) {
        this.mData.remove(i);
        notifyDataSetChanged();
    }

    public void remove(T t) {
        this.mData.remove(t);
        notifyDataSetChanged();
    }

    public void replaceAll(List<T> list) {
        this.mData.clear();
        if (list != null && list.size() > 0) {
            this.mData.addAll(list);
        }
        notifyDataSetChanged();
    }

    public void set(int i, T t) {
        this.mData.set(i, t);
        notifyDataSetChanged();
    }

    public void set(T t, T t2) {
        set(this.mData.indexOf(t), (int) t2);
    }
}
