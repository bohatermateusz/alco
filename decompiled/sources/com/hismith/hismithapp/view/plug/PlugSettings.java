package com.hismith.hismithapp.view.plug;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class PlugSettings {
    private View baseView;
    private Context mContext;

    public PlugSettings(Context context) {
        this.mContext = context;
        this.baseView = LayoutInflater.from(context).inflate(R.layout.plug_settings, (ViewGroup) null, false);
        initView();
        init();
        setListener();
    }

    private void init() {
    }

    private void initView() {
    }

    private void setListener() {
    }

    public View getBaseView() {
        return this.baseView;
    }
}
