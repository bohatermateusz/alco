package com.hismith.hismithapp.receiver;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

/* loaded from: classes3.dex */
public class NetBroadcastReceiver extends BroadcastReceiver {
    public static final int NETWORK_MOBILE = 1;
    public static final int NETWORK_NONE = 0;
    public static final int NETWORK_WIFI = 2;
    public NetChangeListener mListener;

    public interface NetChangeListener {
        void onChangeListener(int i);
    }

    public int getNetWorkState(Context context) {
        NetworkInfo activeNetworkInfo = ((ConnectivityManager) context.getSystemService("connectivity")).getActiveNetworkInfo();
        if (activeNetworkInfo != null && activeNetworkInfo.isConnected()) {
            if (activeNetworkInfo.getType() == 1) {
                return 2;
            }
            if (activeNetworkInfo.getType() == 0) {
                return 1;
            }
        }
        return 0;
    }

    @Override // android.content.BroadcastReceiver
    public void onReceive(Context context, Intent intent) {
        if (intent.getAction().equals("android.net.conn.CONNECTIVITY_CHANGE")) {
            int netWorkState = getNetWorkState(context);
            NetChangeListener netChangeListener = this.mListener;
            if (netChangeListener != null) {
                netChangeListener.onChangeListener(netWorkState);
            }
        }
    }

    public void setNetChangeListener(NetChangeListener netChangeListener) {
        this.mListener = netChangeListener;
    }
}
