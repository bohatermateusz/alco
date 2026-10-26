package com.hismith.hismithapp.receiver;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class BluetoothReceiver extends BroadcastReceiver {
    @Override // android.content.BroadcastReceiver
    public void onReceive(Context context, Intent intent) {
        if (intent.getAction().equals("android.bluetooth.adapter.action.STATE_CHANGED")) {
            int intExtra = intent.getIntExtra("android.bluetooth.adapter.extra.STATE", 0);
            if (intExtra == 12) {
                LogUtil.e("BluetoothAdapter STATE_OFF");
                BleManager.getInstance(context).startLeScan();
                Intent intent2 = new Intent(Constans.RECEIVER_BLUETOOTH_STATE);
                intent2.putExtra("android.bluetooth.adapter.extra.STATE", intExtra);
                context.sendBroadcast(intent2);
                return;
            }
            if (intExtra == 10) {
                LogUtil.e("BluetoothAdapter STATE_OFF");
                BleManager.getInstance(context).disconnectAllGatt();
                Intent intent3 = new Intent(Constans.RECEIVER_BLUETOOTH_STATE);
                intent3.putExtra("android.bluetooth.adapter.extra.STATE", intExtra);
                context.sendBroadcast(intent3);
            }
        }
    }
}
