package com.hismith.hismithapp.fragment;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public abstract class ControlFragment extends LazyLoadFragment {
    private BaseActivity baseActivity;
    private int speedSafe = 100;
    private int vibrateSafe = 100;
    private final int WHAT_SPEED = 1;
    private final int WHAT_MODE = 2;
    private final int WHAT_VIBRATE = 3;
    private final int WHAT_VIBRATE_MODE = 4;
    private final int WHAT_MIC = 5;
    private final int WHAT_SHAKE = 6;
    private Handler uiHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.ControlFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int intValue = ((Integer) message.obj).intValue();
            int i = message.what;
            if (i == 1) {
                ControlFragment.this.refreshSpeedUI(intValue);
                return false;
            }
            if (i == 2) {
                ControlFragment.this.refreshSpeedModeUI(intValue);
                return false;
            }
            if (i == 3) {
                ControlFragment.this.refreshVibrateUI(intValue);
                return false;
            }
            if (i != 4) {
                return false;
            }
            ControlFragment.this.refreshVibrateModeUI(intValue);
            return false;
        }
    });
    BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.ControlFragment.2
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (!Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                SocketManager.RECEIVER_SOCKET_NOTIFY.equals(action);
                return;
            }
            String stringExtra = intent.getStringExtra(Constans.BleType);
            String stringExtra2 = intent.getStringExtra(Constans.BleData0);
            String stringExtra3 = intent.getStringExtra(Constans.BleData1);
            LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
            if ("AA".equals(stringExtra)) {
                ControlFragment.this.initBleAAData(stringExtra2, stringExtra3);
            } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                ControlFragment.this.initBleCCData(stringExtra2, stringExtra3);
            }
        }
    };

    private void init() {
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleAAData(String str, String str2) {
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2) {
        try {
            LogUtil.e("data0:" + str + ", data1:" + str2);
            Message message = new Message();
            message.what = Integer.valueOf(str, 16).intValue();
            message.obj = Integer.valueOf(str2, 16);
            this.uiHandler.sendMessage(message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this.baseActivity, this.mReceiver, intentFilter);
    }

    private void setListener() {
    }

    private void unregisterReceiver() {
        try {
            this.baseActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        unregisterReceiver();
    }

    public abstract void refreshAutoUI(boolean z);

    public abstract void refreshMicUI(boolean z);

    public abstract void refreshMusicUI(boolean z);

    public abstract void refreshPositionMaxUI(int i);

    public abstract void refreshPositionMinUI(int i);

    public abstract void refreshPositionUI(int i);

    public abstract void refreshShakeUI(boolean z);

    public abstract void refreshShowHeart(boolean z);

    public abstract void refreshSmoothUI(int i);

    public abstract void refreshSpeedModeUI(int i);

    public abstract void refreshSpeedSafeUI(int i);

    public abstract void refreshSpeedUI(int i);

    public abstract void refreshVibrateModeUI(int i);

    public abstract void refreshVibrateSafeUI(int i);

    public abstract void refreshVibrateUI(int i);

    public void sendSpeed(int i) {
        BleManager.getInstance(this.baseActivity).writeSpeed(i);
    }

    public void setSpeedSafe(int i) {
        refreshSpeedSafeUI(i);
    }
}
