package com.hismith.hismithapp.fragment;

import android.os.Bundle;
import androidx.fragment.app.Fragment;

/* loaded from: classes3.dex */
public abstract class LazyLoadFragment extends Fragment {
    protected boolean isPrepared = false;
    protected boolean isLoad = false;

    protected abstract void lazyLoad();

    public void loadData() {
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.isLoad = false;
    }

    @Override // androidx.fragment.app.Fragment
    public void onStart() {
        super.onStart();
        this.isPrepared = true;
        if (getUserVisibleHint() && this.isPrepared) {
            if (!this.isLoad) {
                lazyLoad();
            }
            this.isLoad = true;
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void setUserVisibleHint(boolean z) {
        super.setUserVisibleHint(z);
        if (getUserVisibleHint() && this.isPrepared) {
            if (!this.isLoad) {
                lazyLoad();
            }
            this.isLoad = true;
        }
    }
}
